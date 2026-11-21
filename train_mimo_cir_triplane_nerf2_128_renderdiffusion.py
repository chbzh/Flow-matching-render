"""128x128 tri-plane CSI training with a NeRF2-style RF field.

The model predicts clean CSI x0 from noisy CSI xt, an RSU image and point cloud,
four CAV camera images and the CAV point cloud. RX position is used only by the
CSI rendering stage, not by the tri-plane generator. The RF renderer follows NeRF2's ray sampling and
complex signal accumulation, generalized from 26 complex outputs to the full
16x64 MIMO matrix synthesized by this repository's CIR dataset.
"""
import argparse
import datetime
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
import yaml

from channel_matrix_dataset_mimo_camera import MIMOCSIDataset


class MultiSensorCAVCSIDataset(MIMOCSIDataset):
    """RSU sensors plus four CAV cameras and the selected CAV LiDAR cloud."""
    def __init__(self, *args, cav_id, cav_num_points=24000, **kwargs):
        super().__init__(*args, cav_id=cav_id, **kwargs)
        self.cav_num_points = cav_num_points
        cav_pcd_files = {p.stem: p for p in self.rx_dir.glob("*.pcd")}
        kept = []
        for sample in self.samples:
            rsu_camera = self.tx_dir / f"{sample['frame']}_camera0.png"
            cav_pcd = cav_pcd_files.get(sample["frame"])
            if rsu_camera.is_file() and cav_pcd is not None:
                sample["rsu_camera"] = rsu_camera
                sample["cav_pcd"] = cav_pcd
                kept.append(sample)
        self.samples = kept
        if not self.samples:
            raise FileNotFoundError(
                f"No aligned RSU camera, four CAV cameras, CAV LiDAR and CSI for CAV{cav_id}"
            )

    def __getitem__(self, index):
        sample = self.samples[index]
        item = super().__getitem__(index)
        item["rsu_image"] = self._read_camera(sample["rsu_camera"])

        # Transform this CAV's local LiDAR cloud into the shared world frame.
        rx_yaml = self._read_yaml(sample["rx"], sample["frame"], f"cav_{self.cav_id}")
        lidar_pose = rx_yaml["sensors"]["lidar_pose"]
        cloud = o3d.io.read_point_cloud(str(sample["cav_pcd"]))
        cav_points = np.asarray(cloud.points, dtype=np.float32)
        if cav_points.size == 0:
            raise ValueError(f"Empty CAV point cloud: {sample['cav_pcd']}")
        cav_points = self._points_to_world(cav_points, lidar_pose)
        rng = np.random.default_rng(self.seed + index)
        if len(cav_points) >= self.cav_num_points:
            cav_points = cav_points[rng.choice(len(cav_points), self.cav_num_points, replace=False)]
        else:
            cav_points = np.concatenate((
                cav_points,
                cav_points[rng.choice(len(cav_points), self.cav_num_points - len(cav_points), replace=True)],
            ), axis=0)
        item["cav_points"] = torch.from_numpy(np.ascontiguousarray(cav_points))
        item["cav_id"] = self.cav_id
        # MIMOCSIDataset's `images` field contains all four CAV camera views.
        item["cav_images"] = item.pop("images")
        return item

def group_count(channels, maximum=32):
    groups = min(channels, maximum)
    while channels % groups:
        groups -= 1
    return groups


class SinusoidalPositionEncoding(nn.Module):
    """Learned projection of raw xyz plus fixed sinusoidal xyz frequencies."""
    def __init__(self, output_dim=128, max_frequency=128.0):
        super().__init__()
        frequency_count = max(1, math.ceil(max(0, output_dim - 3) / 6))
        self.register_buffer("frequencies", torch.logspace(
            0, math.log10(max_frequency), frequency_count) * math.pi)
        self.proj = nn.Sequential(nn.Linear(3 + 6 * frequency_count, output_dim),
                                  nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, position):
        phase = position.unsqueeze(-1) * self.frequencies
        encoded = torch.cat((position, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
        return self.proj(encoded)


def sinusoidal_timestep(timestep, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=timestep.device, dtype=torch.float32) / max(half - 1, 1))
    phase = timestep.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    result = torch.cat((phase.sin(), phase.cos()), dim=-1)
    return F.pad(result, (0, dim - result.shape[-1]))


class FiLMResidualBlock(nn.Module):
    """Residual U-Net block with global FiLM conditioning."""
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, condition):
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(condition).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(hidden))


class TriPlaneEncoder(nn.Module):
    """Direct CSI-to-triplane FiLM U-Net conditioned on cameras, LiDAR and TX."""
    def __init__(self, plane_dim=64, resolution=32, condition_dim=256,
                 world_extent=100.0, unet_depth=2):
        super().__init__()
        if unet_depth < 1 or unet_depth > 4:
            raise ValueError("unet_depth must be between 1 and 4")
        self.resolution, self.world_extent = resolution, world_extent
        self.unet_depth, self.plane_dim = unet_depth, plane_dim
        self.condition_dim = condition_dim

        # Independent modality encoders. Their spatial tokens are cross-attended
        # by U-Net features instead of being splatted into the output planes.
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 192, 3, 2, 1), nn.GroupNorm(16, 192), nn.GELU())
        self.image_token_projection = nn.Linear(192, condition_dim)
        # Shared weights across the four CAV cameras, separate from the RSU image branch.
        self.cav_image_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 192, 3, 2, 1), nn.GroupNorm(16, 192), nn.GELU())
        self.cav_image_token_projection = nn.Linear(192, condition_dim)
        self.cav_camera_embedding = nn.Parameter(torch.randn(1, 4, condition_dim) * 0.02)
        self.pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.GELU())
        self.point_token_projection = nn.Linear(256, condition_dim)
        # Separate CAV PointNet branch with the same architecture as RSU PointNet.
        self.cav_pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.GELU())
        self.cav_point_token_projection = nn.Linear(256, condition_dim)
        self.tx_position = SinusoidalPositionEncoding(128)
        self.tx_token_projection = nn.Sequential(nn.Linear(128, condition_dim),
                                                 nn.LayerNorm(condition_dim), nn.GELU())
        self.time_projection = nn.Sequential(nn.Linear(128, condition_dim), nn.SiLU(),
                                             nn.Linear(condition_dim, condition_dim))
        self.modality_embedding = nn.Parameter(torch.randn(1, 5, condition_dim) * 0.02)
        self.global_condition = nn.Sequential(
            nn.Linear(5 * condition_dim, condition_dim), nn.LayerNorm(condition_dim), nn.GELU())
        self.diffusion_condition = nn.Sequential(
            nn.Linear(2 * condition_dim, condition_dim), nn.LayerNorm(condition_dim), nn.GELU())

        # CSI (real and imaginary channels) goes directly into the U-Net input
        # convolution: there is no CSI CNN/encoder before the U-Net.
        self.input = nn.Conv2d(2, plane_dim, 3, padding=1)
        widths = [plane_dim * (2 ** level) for level in range(unet_depth + 1)]
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                FiLMResidualBlock(widths[level], widths[level], condition_dim),
                FiLMResidualBlock(widths[level], widths[level], condition_dim),
            ]) for level in range(unet_depth)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv2d(widths[level], widths[level + 1], 3, stride=2, padding=1)
            for level in range(unet_depth)
        ])
        self.middle = nn.ModuleList([
            FiLMResidualBlock(widths[-1], widths[-1], condition_dim),
            FiLMResidualBlock(widths[-1], widths[-1], condition_dim),
        ])
        self.upsamples = nn.ModuleList([
            nn.Conv2d(widths[level + 1], widths[level], 3, padding=1)
            for level in range(unet_depth)
        ])
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                FiLMResidualBlock(2 * widths[level], widths[level], condition_dim),
                FiLMResidualBlock(widths[level], widths[level], condition_dim),
            ]) for level in range(unet_depth)
        ])
        self.output = nn.Sequential(
            nn.GroupNorm(group_count(plane_dim), plane_dim), nn.SiLU(),
            nn.Conv2d(plane_dim, 3 * plane_dim, 3, padding=1))

    def forward(self, noisy_csi, timestep, points, rsu_image, cav_images, cav_points, tx_pos):
        # RSU camera feature map -> 16 tokens.
        image_map = F.adaptive_avg_pool2d(self.image_backbone(rsu_image), (4, 4))
        image_tokens = self.image_token_projection(image_map.flatten(2).transpose(1, 2))

        # Encode four CAV views with one shared CNN; retain one token per view.
        batch, camera_count, channels, image_h, image_w = cav_images.shape
        cav_image_batch = cav_images.reshape(batch * camera_count, channels, image_h, image_w)
        cav_image_map = F.adaptive_avg_pool2d(self.cav_image_backbone(cav_image_batch), (1, 1))
        cav_image_tokens = self.cav_image_token_projection(cav_image_map.flatten(1))
        cav_image_tokens = cav_image_tokens.reshape(batch, camera_count, self.condition_dim)
        if camera_count == 4:
            cav_image_tokens = cav_image_tokens + self.cav_camera_embedding

        # RSU and CAV LiDAR each have an independent PointNet branch.
        point_count = min(points.shape[1], 32)
        point_ids = torch.linspace(0, points.shape[1] - 1, point_count,
                                   device=points.device).long()
        coordinates = ((points - tx_pos[:, None]) / self.world_extent).clamp(-1, 1)
        point_features = self.pointnet(coordinates.transpose(1, 2).contiguous()).transpose(1, 2)
        point_tokens = self.point_token_projection(point_features[:, point_ids])

        cav_point_count = min(cav_points.shape[1], 32)
        cav_point_ids = torch.linspace(0, cav_points.shape[1] - 1, cav_point_count,
                                       device=cav_points.device).long()
        # CAV points are already in the shared world frame. Keep their
        # preprocessing independent of RX position so RX cannot leak into the
        # tri-plane generation branch through coordinate centering.
        cav_coordinates = (cav_points / self.world_extent).clamp(-1, 1)
        cav_point_features = self.cav_pointnet(
            cav_coordinates.transpose(1, 2).contiguous()).transpose(1, 2)
        cav_point_tokens = self.cav_point_token_projection(cav_point_features[:, cav_point_ids])

        tx_pe = self.tx_position(tx_pos / 100.0)
        tx_token = self.tx_token_projection(tx_pe).unsqueeze(1)
        time_embedding = self.time_projection(sinusoidal_timestep(timestep, 128))

        # Pool each modality separately, then fuse once into FiLM vectors.
        # This avoids spatial cross-attention at every U-Net resolution.
        modality_summary = torch.cat((
            (image_tokens + self.modality_embedding[:, 0:1]).mean(dim=1),
            (cav_image_tokens + self.modality_embedding[:, 1:2]).mean(dim=1),
            (point_tokens + self.modality_embedding[:, 2:3]).mean(dim=1),
            (cav_point_tokens + self.modality_embedding[:, 3:4]).mean(dim=1),
            (tx_token + self.modality_embedding[:, 4:5]).squeeze(1),
        ), dim=-1)
        fused = self.global_condition(modality_summary)
        condition = self.diffusion_condition(torch.cat((fused, time_embedding), dim=-1))

        # Direct noisy CSI input, resized only to match the requested plane grid.
        x = F.interpolate(noisy_csi, size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)
        x = self.input(x)
        skips = []
        for blocks, downsample in zip(self.encoders, self.downsamples):
            for block in blocks:
                x = block(x, condition)
            skips.append(x)
            x = downsample(x)
        for block in self.middle:
            x = block(x, condition)
        for level in reversed(range(self.unet_depth)):
            skip = skips[level]
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = self.upsamples[level](x)
            x = torch.cat((x, skip), dim=1)
            for block in self.decoders[level]:
                x = block(x, condition)

        # One U-Net prediction is split into XY, XZ and YZ feature planes.
        planes = self.output(x).reshape(
            noisy_csi.shape[0], 3, self.plane_dim, self.resolution, self.resolution)
        return planes, fused, tx_pe, condition


class NeRF2CSIField(nn.Module):
    """NeRF2-style four-field RF MLP, conditioned on tri-plane query features."""
    def __init__(self, nr, nt, plane_dim=64, condition_dim=256, hidden_dim=256,
                 world_extent=100.0, position_frequencies=6, direction_frequencies=6):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.channels = nr * nt
        self.world_extent = world_extent
        self.position_frequencies, self.direction_frequencies = position_frequencies, direction_frequencies
        # RX position belongs to the CSI rendering path only.
        self.rx_position = SinusoidalPositionEncoding(128)
        pos_dim = 3 * (1 + 2 * position_frequencies)
        view_dim = 3 * (1 + 2 * direction_frequencies)
        in_dim = 3 * plane_dim + pos_dim + view_dim + 2 * condition_dim + 2 * 128
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.skip = nn.Linear(in_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 4 * self.channels)

    @staticmethod
    def positional_encode(value, frequencies):
        bands = (2.0 ** torch.arange(frequencies, device=value.device, dtype=value.dtype)) * math.pi
        phase = value.unsqueeze(-1) * bands
        return torch.cat((value, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)

    @staticmethod
    def sample_planes(planes, coords):
        # coords: [B,V,S,3], normalized TX-relative xyz.
        batch, views, samples, _ = coords.shape
        samples_out = []
        for plane_index, axes in enumerate(((0, 1), (0, 2), (1, 2))):
            grid = coords[..., list(axes)].reshape(batch, views * samples, 1, 2)
            feature = F.grid_sample(planes[:, plane_index], grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            samples_out.append(feature[..., 0].transpose(1, 2).reshape(batch, views, samples, -1))
        return torch.cat(samples_out, dim=-1)

    def forward(self, planes, points, directions, tx_pos, fused, rx_pos, tx_pe, condition):
        # points [B,V,S,3], directions [B,V,3].
        batch, views, samples, _ = points.shape
        local = ((points - tx_pos[:, None, None, :]) / self.world_extent).clamp(-1, 1)
        plane_features = self.sample_planes(planes, local)
        pos = self.positional_encode(local, self.position_frequencies)
        view = self.positional_encode(directions[:, :, None].expand(-1, -1, samples, -1),
                                      self.direction_frequencies)
        rx_pe = self.rx_position(rx_pos / 100.0)
        contexts = [fused[:, None, None].expand(-1, views, samples, -1),
                    rx_pe[:, None, None].expand(-1, views, samples, -1),
                    tx_pe[:, None, None].expand(-1, views, samples, -1),
                    condition[:, None, None].expand(-1, views, samples, -1)]
        inputs = torch.cat((plane_features, pos, view, *contexts), dim=-1)
        hidden = self.trunk(inputs) + self.skip(inputs)
        return self.output(F.relu(hidden))


class NeRF2CSIRenderer(nn.Module):
    """NeRF2 CSI rendering: ray sample -> RF fields -> coherent path/ray sum."""
    def __init__(self, nr, nt, near=0.0, far=100.0, n_samples=32,
                 carrier_hz=2.4e9, ray_chunk=4, elevation_count=9, azimuth_count=36):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.near, self.far, self.n_samples = near, far, n_samples
        self.carrier_hz, self.ray_chunk = carrier_hz, ray_chunk
        # Match CSI_dataset.gen_rays_gateways() in NeRF2: 9 elevations (10..90°)
        # by 36 azimuths (0..350°), with rays originating at the base station.
        beta = torch.linspace(10.0, 90.0, elevation_count) * math.pi / 180.0
        alpha = torch.linspace(0.0, 350.0, azimuth_count) * math.pi / 180.0
        el, az = torch.meshgrid(beta, alpha, indexing="ij")
        directions = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), dim=-1)
        self.register_buffer("direction_grid", directions.reshape(-1, 3))

    def render(self, field, planes, tx_pos, rx_pos, fused, tx_pe, condition):
        batch = tx_pos.shape[0]
        directions_all = self.direction_grid.to(tx_pos).unsqueeze(0).expand(batch, -1, -1)
        view_count = directions_all.shape[1]
        t_vals = torch.linspace(self.near, self.far, self.n_samples,
                                device=tx_pos.device, dtype=tx_pos.dtype)
        wavelength = 299792458.0 / self.carrier_hz
        accumulated = torch.zeros(batch, self.nr * self.nt, device=tx_pos.device,
                                  dtype=torch.complex64 if tx_pos.dtype == torch.float32 else torch.complex128)
        for start in range(0, view_count, self.ray_chunk):
            directions = directions_all[:, start:start + self.ray_chunk]
            views = directions.shape[1]
            depths = t_vals.view(1, 1, -1).expand(batch, views, -1)
            points = tx_pos[:, None, None, :] + directions[:, :, None, :] * depths[..., None]
            raw = field(planes, points, directions, tx_pos, fused, rx_pos, tx_pe, condition)
            raw = raw.reshape(batch, views, self.n_samples, 4, self.nr * self.nt)
            att_a, att_p, sig_a, sig_p = raw.unbind(dim=3)
            att_a = F.leaky_relu(att_a).abs()
            sig_a = F.leaky_relu(sig_a).abs()
            att_p = torch.sigmoid(att_p) * (2 * math.pi) - math.pi
            sig_p = torch.sigmoid(sig_p) * (2 * math.pi) - math.pi

            delta = t_vals[1:] - t_vals[:-1]
            # NeRF2 uses a large terminal interval. Keep its same accumulated
            # attenuation behavior, but cap the last interval at one sample step
            # to avoid an infinite sentinel contaminating mixed precision.
            delta = torch.cat((delta, delta[-1:])).view(1, 1, -1, 1)
            delta = delta * torch.linalg.vector_norm(directions, dim=-1)[:, :, None, None]
            amplitude = torch.exp(torch.cumsum(-att_a * delta, dim=2))
            phase = torch.exp(1j * torch.cumsum(
                att_p + (2 * math.pi / wavelength) * delta, dim=2))
            signal_phase = torch.exp(1j * sig_p)
            ray_signal = torch.sum(sig_a * signal_phase * amplitude * phase, dim=2)
            accumulated = accumulated + ray_signal.sum(dim=1)
        return accumulated.reshape(batch, self.nr, self.nt)


class TriPlaneNeRF2Diffusion(nn.Module):
    def __init__(self, nr, nt, plane_dim=64, plane_resolution=128, world_extent=100.0,
                 near=0, far=100, n_samples=32, carrier_hz=2.4e9, ray_chunk=4,
                 unet_depth=2):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.encoder = TriPlaneEncoder(plane_dim=plane_dim, resolution=plane_resolution,
                                       world_extent=world_extent, unet_depth=unet_depth)
        self.field = NeRF2CSIField(nr, nt, plane_dim=plane_dim, world_extent=world_extent)
        self.renderer = NeRF2CSIRenderer(nr, nt, near, far, n_samples,
                                         carrier_hz=carrier_hz, ray_chunk=ray_chunk)
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, 1000))
        self.register_buffer("alpha_bars", torch.cumprod(1 - self.betas, dim=0))

    def render_x0(self, noisy_csi, timestep, batch):
        planes, fused, tx_pe, condition = self.encoder(
            noisy_csi, timestep, batch["points"], batch["rsu_image"],
            batch["cav_images"], batch["cav_points"], batch["tx_pos"])
        complex_csi = self.renderer.render(self.field, planes, batch["tx_pos"], batch["rx_pos"],
                                           fused, tx_pe, condition)
        # The network is trained on per-sample max-normalized CSI, matching the
        # existing CIR dataset training convention.
        complex_csi = complex_csi / complex_csi.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return torch.stack((complex_csi.real, complex_csi.imag), dim=1)

    def training_loss(self, batch, target):
        batch_size = target.shape[0]
        timestep = torch.randint(0, self.betas.numel(), (batch_size,), device=target.device)
        noise = torch.randn_like(target)
        alpha_bar = self.alpha_bars[timestep].view(batch_size, 1, 1, 1)
        noisy = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
        prediction = self.render_x0(noisy, timestep, batch)
        return F.mse_loss(prediction, target), prediction

    @torch.inference_mode()
    def sample(self, batch, steps=20):
        batch_size = batch["points"].shape[0]
        noisy = torch.randn(batch_size, 2, self.nr, self.nt, device=batch["points"].device)
        times = torch.linspace(self.betas.numel() - 1, 0, steps, device=noisy.device).long().unique_consecutive()
        for index, current in enumerate(times):
            timestep = torch.full((batch_size,), int(current), device=noisy.device, dtype=torch.long)
            prediction = self.render_x0(noisy, timestep, batch)
            alpha_current = self.alpha_bars[current]
            epsilon = (noisy - alpha_current.sqrt() * prediction) / (1 - alpha_current).sqrt().clamp_min(1e-5)
            if index + 1 == len(times):
                noisy = prediction
            else:
                alpha_previous = self.alpha_bars[times[index + 1]]
                noisy = alpha_previous.sqrt() * prediction + (1 - alpha_previous).sqrt() * epsilon
        return noisy


def load_render_config(config_path):
    if not config_path or not Path(config_path).is_file():
        return {}
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return config.get("render", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-root", default="/home/largeDisk/bingzheng/DGS/WRF-GSplus-main/MMData/CSI_sensor_data")
    parser.add_argument("--channel-root", default="/home/largeDisk/bingzheng/DGS/WRF-GSplus-main/MMData/CSI_data")
    parser.add_argument("--scenario", default="Town03_Tjunction_wiz_slope_seed42")
    parser.add_argument("--channel-scenario", default="Town03_Tjunction")
    parser.add_argument("--rx-pose-key", choices=("vehicle_pose", "lidar_pose"), default="vehicle_pose")
    parser.add_argument("--tx-pose-key", choices=("rsu_pose", "lidar_pose"), default="rsu_pose")
    parser.add_argument("--label", choices=("channel", "channel_H"), default="channel")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=1024,
                        help="Number of points sampled from the RSU point cloud")
    parser.add_argument("--cav-num-points", type=int, default=24000,
                        help="Number of points sampled from each CAV point cloud")
    parser.add_argument("--plane-resolution", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, choices=(1, 2, 3, 4), default=4,
                        help="Number of U-Net downsampling stages")
    parser.add_argument("--plane-dim", type=int, default=64)
    parser.add_argument("--world-extent", type=float, default=100.0)
    parser.add_argument("--nerf2-config", default="/home/largeDisk/bingzheng/NeRF2-main/configs/mimo-csi.yml")
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--carrier-hz", type=float, default=2.4e9)
    parser.add_argument("--ray-chunk", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-start-epoch", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-every", type=int, default=20)
    parser.add_argument("--inference-steps", type=int, default=1) #  根据renderdiffusion的描述，测试就只执行一次迭代
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency-offset-hz", type=float, default=0.0)
    parser.add_argument("--output", default="runs/mimo_csi_triplane_nerf2_128_renderdiffusion")
    args = parser.parse_args()
    config_render = load_render_config(args.nerf2_config)
    # NeRF2's config range is in that dataset's coordinate scale (far=5).
    # This dataset keeps CARLA world positions in meters, so default to 100 m;
    # set --near/--far to reuse another scene's scale explicitly.
    near = 0.0 if args.near is None else args.near
    far = 100.0 if args.far is None else args.far
    n_samples = config_render.get("n_samples", 32) if args.n_samples is None else args.n_samples
    if (far <= near or n_samples < 2 or args.batch_size < 1 or args.ray_chunk < 1
            or args.num_points < 1 or args.cav_num_points < 1):
        raise ValueError("Require far > near, n_samples >= 2, and positive batch, point, and ray chunk sizes")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cav_datasets = [MultiSensorCAVCSIDataset(
        args.sensor_root, args.channel_root, args.scenario, args.channel_scenario,
        cav_id=cav_id, cav_num_points=args.cav_num_points,
        num_points=args.num_points, frequency_offset_hz=args.frequency_offset_hz,
        seed=args.seed, rx_pose_key=args.rx_pose_key, tx_pose_key=args.tx_pose_key,
        image_size=args.image_size)
        for cav_id in (1, 2, 3, 4)]
    dataset = ConcatDataset(cav_datasets)
    # Keep every CAV measurement from a frame in the same split.
    frame_to_indices = {}
    dataset_offset = 0
    for cav_dataset in cav_datasets:
        for local_index, sample_info in enumerate(cav_dataset.samples):
            frame_to_indices.setdefault(sample_info["frame"], []).append(dataset_offset + local_index)
        dataset_offset += len(cav_dataset)
    frame_ids = sorted(frame_to_indices, key=lambda frame: int(frame))
    if len(frame_ids) < 2:
        raise ValueError("At least two frame-aligned samples are required")
    shuffled_frames = torch.randperm(
        len(frame_ids), generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_count = max(1, min(len(frame_ids) - 1, round(len(frame_ids) * args.val_fraction)))
    val_frames = {frame_ids[index] for index in shuffled_frames[:val_count]}
    train_ids = [index for frame, indices in frame_to_indices.items()
                 if frame not in val_frames for index in indices]
    val_ids = [index for frame, indices in frame_to_indices.items()
               if frame in val_frames for index in indices]
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    first_cav_dataset = cav_datasets[0]
    first_train_path = next(
        sample_info["cir"] for cav_dataset in cav_datasets for sample_info in cav_dataset.samples
        if sample_info["frame"] not in val_frames)
    sample = first_cav_dataset._read_channels(first_train_path)[
        0 if args.label == "channel" else 1]
    nr, nt = sample.shape[-2:]
    model = TriPlaneNeRF2Diffusion(nr, nt, args.plane_dim, args.plane_resolution,
        args.world_extent, near, far, n_samples, args.carrier_hz, args.ray_chunk,
        args.unet_depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=args.lr_gamma)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) / timestamp
    log_dir = Path("logs") / f"mimo_csi_triplane_nerf2_128_renderdiffusion_{timestamp}"
    output.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"triplane_nerf2_{timestamp}")
    logger.setLevel(logging.INFO); logger.propagate = False
    handler = logging.FileHandler(log_dir / "logger.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(handler)
    logger.info("CSI shape=%dx%d label=%s CAVs=1-4 train=%d val=%d RSU_points=%d CAV_points=%d rays=%d samples=%d range=[%g,%g]",
                nr, nt, args.label, len(train_ids), len(val_ids), args.num_points,
                args.cav_num_points, model.renderer.direction_grid.shape[0], n_samples, near, far)

    def move(batch):
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def target_from(batch):
        target = batch[args.label].float()
        scale = torch.linalg.vector_norm(target, dim=1).amax(dim=(-2, -1), keepdim=True).unsqueeze(1)
        return target / scale.clamp_min(1e-8)

    def metric(pred, target):
        p, t = pred.flatten(1), target.flatten(1)
        cosine = F.cosine_similarity(p, t, dim=1)
        nmse = (p-t).square().sum(1) / t.square().sum(1).clamp_min(1e-12)
        return cosine, 10 * torch.log10(nmse.clamp_min(1e-12))

    def evaluate(epoch):
        model.eval(); cos_all, nmse_all = [], []
        save_dir = log_dir / "test" / str(epoch); image_dir = save_dir / "pred_csi"
        image_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "all_cos_nmse.txt").open("w", encoding="utf-8") as metrics_file, torch.inference_mode():
            metrics_file.write("index\tframe_id\tcosine_similarity\tnmse_db\n")
            sample_no = 0
            for raw in tqdm(val_loader, desc=f"Test {epoch}", leave=False):
                batch = move(raw); target = target_from(batch)
                pred = model.sample(batch, args.inference_steps)
                cos, nmse = metric(pred, target)
                cos_all.extend(cos.cpu().tolist()); nmse_all.extend(nmse.cpu().tolist())
                for i in range(pred.shape[0]):
                    frame = str(batch["frame_id"][i]); stem = f"{sample_no:05d}_{frame}"
                    metrics_file.write(f"{sample_no}\t{frame}\t{cos[i].item():.8f}\t{nmse[i].item():.8f}\n")
                    pn, tn = pred[i].cpu().numpy(), target[i].cpu().numpy()
                    np.savez_compressed(save_dir / f"{stem}.npz", prediction=pn, ground_truth=tn,
                                        frame_id=frame, label=args.label)
                    pm, tm = np.hypot(pn[0], pn[1]), np.hypot(tn[0], tn[1])
                    low, high = min(float(pm.min()), float(tm.min())), max(float(pm.max()), float(tm.max()))
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    img = ax[0].imshow(tm, origin="lower", aspect="auto", cmap="viridis", vmin=low, vmax=max(high, low+1e-12))
                    ax[0].set_title("Ground truth CSI |H|")
                    ax[1].imshow(pm, origin="lower", aspect="auto", cmap="viridis", vmin=low, vmax=max(high, low+1e-12))
                    ax[1].set_title("NeRF2-rendered CSI |H|")
                    fig.subplots_adjust(bottom=0.2, wspace=0.12)
                    colorbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.035])
                    fig.colorbar(img, cax=colorbar_ax, orientation="horizontal")
                    fig.savefig(image_dir / f"{stem}.png", dpi=150); plt.close(fig)
                    sample_no += 1
        model.train()
        return float(np.mean(cos_all)), float(np.mean(nmse_all))

    history, best_nmse = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); losses, cos_values, nmse_values = [], [], []
        progress = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
        for raw in progress:
            batch = move(raw); target = target_from(batch)
            loss, pred = model.training_loss(batch, target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cos, nmse = metric(pred.detach(), target)
            losses.append(loss.item()); cos_values.extend(cos.cpu().tolist()); nmse_values.extend(nmse.cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.5g}", cosine=f"{cos.mean().item():.4f}",
                                 nmse_db=f"{nmse.mean().item():.2f}")
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "train_cosine": float(np.mean(cos_values)), "train_nmse_db": float(np.mean(nmse_values)),
                  "lr": optimizer.param_groups[0]["lr"]}
        print(json.dumps(record))
        if epoch % args.test_every == 0 or epoch == args.epochs:
            test_cos, test_nmse = evaluate(epoch)
            record.update(test_cosine=test_cos, test_nmse_db=test_nmse)
            print(f"Epoch {epoch}: test cosine={test_cos:.5f}, NMSE={test_nmse:.3f} dB")
            logger.info("Epoch %d test cosine=%.6f NMSE=%.3f dB", epoch, test_cos, test_nmse)
            if test_nmse < best_nmse:
                best_nmse = test_nmse
                state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "args": vars(args), "nr": nr, "nt": nt,
                         "target_label": args.label}
                torch.save(state, output / f"checkpoint_depth{args.unet_depth}_best.pt")
        if epoch >= args.lr_start_epoch and (epoch - args.lr_start_epoch) % args.lr_step_size == 0:
            scheduler.step(); record["lr"] = optimizer.param_groups[0]["lr"]
        history.append(record)
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "args": vars(args), "nr": nr, "nt": nt,
                     "target_label": args.label}
            torch.save(state, output / f"checkpoint_depth{args.unet_depth}_{epoch}.pt")
            torch.save(state, output / f"checkpoint_depth{args.unet_depth}_latest.pt")
    handler.close(); logger.removeHandler(handler)


if __name__ == "__main__":
    main()
