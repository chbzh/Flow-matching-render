"""Conditional flow-matching tri-plane CSI model with seven sensor inputs.

Five camera observations and two LiDAR point clouds are encoded by modality-
shared encoders. Two conditional flow-matching models generate one tri-plane
space per observation from Gaussian noise. A learned softmax weighted average
of those seven generated spaces is rendered to CSI using an RX-origin NeRF2-
style ray integral.

There is no ground-truth tri-plane in the dataset. During training, modality-
specific posterior encoders use the input modality and target CSI to define
endpoint latent distributions for conditional flow matching. At inference,
those posterior encoders are not used: the flow models generate spaces from
Gaussian noise conditioned only on sensor observations.
"""
import argparse
import datetime
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from train_mimo_cir_triplane_nerf2_128_renderdiffusion import (
    MultiSensorCAVCSIDataset,
    NeRF2CSIField,
    SinusoidalPositionEncoding,
)


def group_count(channels, maximum=32):
    groups = min(channels, maximum)
    while channels % groups:
        groups -= 1
    return groups


def sinusoidal_time(t, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    phase = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    emb = torch.cat((phase.sin(), phase.cos()), dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1]))


class ImageConditionEncoder(nn.Module):
    """Shared camera encoder, used for all four CAV cameras and the RSU camera."""
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 192, 3, 2, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(192, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, images):
        # images: [B, M, 3, H, W]
        batch, count, channels, height, width = images.shape
        encoded = self.net(images.reshape(batch * count, channels, height, width))
        return encoded.reshape(batch, count, -1)


class PointCloudConditionEncoder(nn.Module):
    """Shared PointNet-style encoder for the CAV and RSU LiDAR clouds."""
    def __init__(self, dim=256, world_extent=100.0):
        super().__init__()
        self.world_extent = world_extent
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.GELU())
        self.projection = nn.Sequential(
            nn.Linear(512, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, points):
        # Points are already expressed in the shared world frame.
        xyz = (points / self.world_extent).clamp(-1, 1)
        features = self.point_mlp(xyz.transpose(1, 2).contiguous())
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=-1)
        return self.projection(pooled)


class CSIContextEncoder(nn.Module):
    """Training-only summary of target CSI for posterior latent distributions."""
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, dim), nn.GELU())

    def forward(self, csi):
        return self.net(csi)


class PosteriorSpaceEncoder(nn.Module):
    """Training-time q(z|modality, CSI) that supplies FM endpoint samples."""
    def __init__(self, condition_dim=256, csi_dim=128, space_channels=96,
                 latent_resolution=16):
        super().__init__()
        self.latent_resolution = latent_resolution
        self.base_resolution = min(4, latent_resolution)
        self.base_channels = 128
        self.project = nn.Sequential(
            nn.Linear(condition_dim + csi_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, self.base_channels * self.base_resolution ** 2), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Conv2d(self.base_channels, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 2 * space_channels, 3, padding=1))

    def forward(self, condition, csi_context):
        x = torch.cat((condition, csi_context), dim=-1)
        x = self.project(x).reshape(-1, self.base_channels,
                                    self.base_resolution, self.base_resolution)
        x = F.interpolate(x, size=(self.latent_resolution, self.latent_resolution),
                          mode="bilinear", align_corners=False)
        mean, logvar = self.decoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-8.0, 2.0)


class FlowResidualBlock(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.film = nn.Linear(condition_dim, 2 * channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(condition)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.conv2(F.silu(h)) * 0.5


class ConditionalFlowMatching(nn.Module):
    """Conditional vector field for Gaussian-to-tri-plane latent transport."""
    def __init__(self, space_channels, condition_dim=256, width=96):
        super().__init__()
        self.time_projection = nn.Sequential(
            nn.Linear(128, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, condition_dim), nn.LayerNorm(condition_dim), nn.GELU())
        self.input = nn.Conv2d(space_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList([FlowResidualBlock(width, condition_dim) for _ in range(4)])
        self.output = nn.Sequential(
            nn.GroupNorm(group_count(width), width), nn.SiLU(),
            nn.Conv2d(width, space_channels, 3, padding=1))

    def forward(self, z_t, t, condition):
        cond = self.condition_projection(condition) + self.time_projection(sinusoidal_time(t, 128))
        x = self.input(z_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.output(x)

    def flow_matching_loss(self, target, condition):
        z0 = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device)
        t_view = t[:, None, None, None]
        z_t = (1 - t_view) * z0 + t_view * target
        target_velocity = target - z0
        predicted_velocity = self(z_t, t, condition)
        return F.mse_loss(predicted_velocity, target_velocity)

class RxOriginCSIRenderer(nn.Module):
    """NeRF2-style ray integration with RX as the ray origin for CSI rendering."""
    def __init__(self, nr, nt, near=0.0, far=100.0, n_samples=32,
                 carrier_hz=2.4e9, ray_chunk=4, elevation_count=9,
                 azimuth_count=36, latent_resolution=16):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.near, self.far, self.n_samples = near, far, n_samples
        self.carrier_hz, self.ray_chunk = carrier_hz, ray_chunk
        self.latent_resolution = latent_resolution
        beta = torch.linspace(10.0, 90.0, elevation_count) * math.pi / 180.0
        alpha = torch.linspace(0.0, 350.0, azimuth_count) * math.pi / 180.0
        el, az = torch.meshgrid(beta, alpha, indexing="ij")
        dirs = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), dim=-1)
        self.register_buffer("direction_grid", dirs.reshape(-1, 3))

    def render(self, field, planes, tx_pos, rx_pos, fused, tx_pe, condition):
        batch = rx_pos.shape[0]
        # Triplane features are generated at lower latent resolution for memory
        # efficiency, then resized before spatial queries.
        planes = F.interpolate(
            planes.reshape(batch * 3, planes.shape[2], *planes.shape[-2:]),
            size=(self.latent_resolution, self.latent_resolution), mode="bilinear",
            align_corners=False).reshape(batch, 3, planes.shape[2],
                                         self.latent_resolution, self.latent_resolution)
        directions_all = self.direction_grid.to(rx_pos).unsqueeze(0).expand(batch, -1, -1)
        t_vals = torch.linspace(self.near, self.far, self.n_samples,
                                device=rx_pos.device, dtype=rx_pos.dtype)
        wavelength = 299792458.0 / self.carrier_hz
        accumulated = torch.zeros(batch, self.nr * self.nt, device=rx_pos.device,
                                  dtype=torch.complex64 if rx_pos.dtype == torch.float32 else torch.complex128)
        for start in range(0, directions_all.shape[1], self.ray_chunk):
            directions = directions_all[:, start:start + self.ray_chunk]
            views = directions.shape[1]
            depths = t_vals.view(1, 1, -1).expand(batch, views, -1)
            # RX anchors rendering rays. TX remains an explicit field condition.
            points = rx_pos[:, None, None, :] + directions[:, :, None, :] * depths[..., None]
            raw = field(planes, points, directions, tx_pos, fused, rx_pos, tx_pe, condition)
            raw = raw.reshape(batch, views, self.n_samples, 4, self.nr * self.nt)
            att_a, att_p, sig_a, sig_p = raw.unbind(dim=3)
            att_a = F.leaky_relu(att_a).abs()
            sig_a = F.leaky_relu(sig_a).abs()
            att_p = torch.sigmoid(att_p) * (2 * math.pi) - math.pi
            sig_p = torch.sigmoid(sig_p) * (2 * math.pi) - math.pi
            delta = t_vals[1:] - t_vals[:-1]
            delta = torch.cat((delta, delta[-1:])).view(1, 1, -1, 1)
            delta = delta * torch.linalg.vector_norm(directions, dim=-1)[:, :, None, None]
            attenuation = torch.exp(torch.cumsum(-att_a * delta, dim=2))
            propagation_phase = torch.exp(1j * torch.cumsum(
                att_p + (2 * math.pi / wavelength) * delta, dim=2))
            emitted = sig_a * torch.exp(1j * sig_p)
            ray_signal = torch.sum(emitted * attenuation * propagation_phase, dim=2)
            accumulated = accumulated + ray_signal.sum(dim=1)
        return accumulated.reshape(batch, self.nr, self.nt)


class SevenSensorFlowModel(nn.Module):
    """Two shared conditional FMs generate seven spaces, then softmax-average."""
    def __init__(self, nr, nt, plane_dim=32, latent_resolution=16,
                 render_resolution=64, world_extent=100.0, near=0.0,
                 far=100.0, n_samples=32, carrier_hz=2.4e9, ray_chunk=4,
                 condition_dim=256):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.plane_dim, self.latent_resolution = plane_dim, latent_resolution
        self.render_resolution = render_resolution
        self.space_channels = 3 * plane_dim
        self.image_encoder = ImageConditionEncoder(condition_dim)
        self.lidar_encoder = PointCloudConditionEncoder(condition_dim, world_extent)
        self.csi_context = CSIContextEncoder(128)
        self.slot_embedding = nn.Parameter(torch.randn(7, condition_dim) * 0.02)
        self.image_posterior = PosteriorSpaceEncoder(
            condition_dim, 128, self.space_channels, latent_resolution)
        self.lidar_posterior = PosteriorSpaceEncoder(
            condition_dim, 128, self.space_channels, latent_resolution)
        self.image_flow = ConditionalFlowMatching(self.space_channels, condition_dim)
        self.lidar_flow = ConditionalFlowMatching(self.space_channels, condition_dim)
        self.weight_head = nn.Sequential(nn.Linear(condition_dim, condition_dim // 2),
                                         nn.GELU(), nn.Linear(condition_dim // 2, 1))
        self.render_condition = nn.Sequential(nn.Linear(condition_dim, condition_dim),
                                              nn.LayerNorm(condition_dim), nn.GELU())
        self.tx_position = SinusoidalPositionEncoding(128)
        self.field = NeRF2CSIField(nr, nt, plane_dim=plane_dim,
                                   condition_dim=condition_dim, world_extent=world_extent)
        self.renderer = RxOriginCSIRenderer(
            nr, nt, near, far, n_samples, carrier_hz, ray_chunk,
            latent_resolution=render_resolution)

    def encode_modalities(self, batch):
        cav_images = batch["cav_images"]
        rsu_image = batch["rsu_image"].unsqueeze(1)
        image_inputs = torch.cat((cav_images, rsu_image), dim=1)
        image_cond = self.image_encoder(image_inputs)
        cav_lidar_cond = self.lidar_encoder(batch["cav_points"])
        rsu_lidar_cond = self.lidar_encoder(batch["points"])
        lidar_cond = torch.stack((cav_lidar_cond, rsu_lidar_cond), dim=1)

        # Seven positions: four CAV cameras, CAV LiDAR, RSU camera, RSU LiDAR.
        ordered_cond = torch.cat((image_cond[:, :4], lidar_cond[:, :1],
                                  image_cond[:, 4:5], lidar_cond[:, 1:2]), dim=1)
        ordered_cond = ordered_cond + self.slot_embedding.unsqueeze(0)
        return image_cond, lidar_cond, ordered_cond

    def posterior_endpoints(self, image_cond, lidar_cond, target):
        batch = target.shape[0]
        csi_cond = self.csi_context(target)
        image_slots = torch.tensor([0, 1, 2, 3, 5], device=target.device)
        lidar_slots = torch.tensor([4, 6], device=target.device)
        image_cond = image_cond + self.slot_embedding[image_slots].unsqueeze(0)
        lidar_cond = lidar_cond + self.slot_embedding[lidar_slots].unsqueeze(0)

        image_flat = image_cond.reshape(batch * 5, -1)
        lidar_flat = lidar_cond.reshape(batch * 2, -1)
        image_csi = csi_cond[:, None].expand(-1, 5, -1).reshape(batch * 5, -1)
        lidar_csi = csi_cond[:, None].expand(-1, 2, -1).reshape(batch * 2, -1)
        image_mean, image_logvar = self.image_posterior(image_flat, image_csi)
        lidar_mean, lidar_logvar = self.lidar_posterior(lidar_flat, lidar_csi)
        image_z = image_mean + torch.exp(0.5 * image_logvar) * torch.randn_like(image_mean)
        lidar_z = lidar_mean + torch.exp(0.5 * lidar_logvar) * torch.randn_like(lidar_mean)
        kl = 0.5 * (image_mean.square() + image_logvar.exp() - 1 - image_logvar).mean()
        kl = kl + 0.5 * (lidar_mean.square() + lidar_logvar.exp() - 1 - lidar_logvar).mean()
        image_fm = self.image_flow.flow_matching_loss(image_z, image_flat)
        lidar_fm = self.lidar_flow.flow_matching_loss(lidar_z, lidar_flat)
        return image_fm, lidar_fm, kl

    @staticmethod
    def integrate_flow(flow, condition, steps, space_channels, resolution):
        z = torch.randn(condition.shape[0], space_channels, resolution, resolution,
                        device=condition.device, dtype=condition.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((condition.shape[0],), step / steps,
                           device=condition.device, dtype=condition.dtype)
            z = z + dt * flow(z, t, condition)
        return z

    def generate_spaces(self, image_cond, lidar_cond, ordered_cond, steps):
        batch = image_cond.shape[0]
        image_slots = torch.tensor([0, 1, 2, 3, 5], device=image_cond.device)
        lidar_slots = torch.tensor([4, 6], device=image_cond.device)
        image_condition = (image_cond + self.slot_embedding[image_slots].unsqueeze(0)).reshape(batch * 5, -1)
        lidar_condition = (lidar_cond + self.slot_embedding[lidar_slots].unsqueeze(0)).reshape(batch * 2, -1)
        image_z = self.integrate_flow(self.image_flow, image_condition, steps,
                                      self.space_channels, self.latent_resolution)
        lidar_z = self.integrate_flow(self.lidar_flow, lidar_condition, steps,
                                      self.space_channels, self.latent_resolution)
        image_z = image_z.reshape(batch, 5, 3, self.plane_dim,
                                  self.latent_resolution, self.latent_resolution)
        lidar_z = lidar_z.reshape(batch, 2, 3, self.plane_dim,
                                  self.latent_resolution, self.latent_resolution)
        # CAV cameras 0..3, CAV LiDAR, RSU camera, RSU LiDAR.
        spaces = torch.stack((image_z[:, 0], image_z[:, 1], image_z[:, 2], image_z[:, 3],
                              lidar_z[:, 0], image_z[:, 4], lidar_z[:, 1]), dim=1)
        weights = torch.softmax(self.weight_head(ordered_cond).squeeze(-1), dim=1)
        final_space = (spaces * weights[:, :, None, None, None, None]).sum(dim=1)
        fused = (ordered_cond * weights[:, :, None]).sum(dim=1)
        return final_space, weights, fused

    def render(self, final_space, fused, batch):
        planes = final_space.reshape(final_space.shape[0], 3, self.plane_dim,
                                     self.latent_resolution,
                                     self.latent_resolution)
        render_cond = self.render_condition(fused)
        tx_pe = self.tx_position(batch["tx_pos"] / 100.0)
        complex_csi = self.renderer.render(
            self.field, planes, batch["tx_pos"], batch["rx_pos"], fused, tx_pe, render_cond)
        complex_csi = complex_csi / complex_csi.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return torch.stack((complex_csi.real, complex_csi.imag), dim=1)

    def forward(self, batch, flow_steps, target=None):
        image_cond, lidar_cond, ordered_cond = self.encode_modalities(batch)
        fm_image = fm_lidar = kl = None
        if target is not None:
            fm_image, fm_lidar, kl = self.posterior_endpoints(image_cond, lidar_cond, target)
        final_space, weights, fused = self.generate_spaces(
            image_cond, lidar_cond, ordered_cond, flow_steps)
        prediction = self.render(final_space, fused, batch)
        return prediction, weights, (fm_image, fm_lidar, kl)


def target_from(batch, label):
    target = batch[label].float()
    scale = torch.linalg.vector_norm(target, dim=1).amax(dim=(-2, -1), keepdim=True).unsqueeze(1)
    return target / scale.clamp_min(1e-8)


def metric(pred, target):
    p, t = pred.flatten(1), target.flatten(1)
    cosine = F.cosine_similarity(p, t, dim=1)
    nmse = (p - t).square().sum(1) / t.square().sum(1).clamp_min(1e-12)
    return cosine, 10 * torch.log10(nmse.clamp_min(1e-12))


def make_loaders(args, device):
    cav_datasets = [MultiSensorCAVCSIDataset(
        args.sensor_root, args.channel_root, args.scenario, args.channel_scenario,
        cav_id=cav_id, cav_num_points=args.cav_num_points,
        num_points=args.num_points, frequency_offset_hz=args.frequency_offset_hz,
        seed=args.seed, rx_pose_key=args.rx_pose_key, tx_pose_key=args.tx_pose_key,
        image_size=args.image_size) for cav_id in (1, 2, 3, 4)]
    dataset = ConcatDataset(cav_datasets)
    frame_to_indices, offset = {}, 0
    for cav_dataset in cav_datasets:
        for local_index, sample_info in enumerate(cav_dataset.samples):
            frame_to_indices.setdefault(sample_info["frame"], []).append(offset + local_index)
        offset += len(cav_dataset)
    frame_ids = sorted(frame_to_indices, key=lambda frame: int(frame))
    if len(frame_ids) < 2:
        raise ValueError("At least two frame-aligned samples are required")
    permutation = torch.randperm(len(frame_ids), generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_count = max(1, min(len(frame_ids) - 1, round(len(frame_ids) * args.val_fraction)))
    val_frames = {frame_ids[i] for i in permutation[:val_count]}
    train_ids = [idx for frame, ids in frame_to_indices.items() if frame not in val_frames for idx in ids]
    val_ids = [idx for frame, ids in frame_to_indices.items() if frame in val_frames for idx in ids]
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True, **common)
    val_loader = DataLoader(Subset(dataset, val_ids), shuffle=False, **common)
    return cav_datasets, train_loader, val_loader, train_ids, val_ids, val_frames


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
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--cav-num-points", type=int, default=24000)
    parser.add_argument("--plane-dim", type=int, default=32)
    parser.add_argument("--latent-resolution", type=int, default=16)
    parser.add_argument("--render-resolution", type=int, default=64)
    parser.add_argument("--world-extent", type=float, default=100.0)
    parser.add_argument("--near", type=float, default=0.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--carrier-hz", type=float, default=2.4e9)
    parser.add_argument("--ray-chunk", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-start-epoch", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--fm-train-steps", type=int, default=4)
    parser.add_argument("--fm-inference-steps", type=int, default=8)
    parser.add_argument("--fm-loss-weight", type=float, default=0.1)
    parser.add_argument("--posterior-kl-weight", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency-offset-hz", type=float, default=0.0)
    parser.add_argument("--output", default="runs/mimo_csi_triplane_nerf2_128_renderdiffusion_flowmatching")
    args = parser.parse_args()
    if (args.near < 0 or args.far <= args.near or args.n_samples < 2
            or args.latent_resolution < 4 or args.render_resolution < 4
            or args.fm_train_steps < 1 or args.fm_inference_steps < 1):
        raise ValueError("Invalid ray range, latent resolution, sample count, or flow step count")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, train_loader, val_loader, train_ids, val_ids, val_frames = make_loaders(args, device)
    first_train_path = next(info["cir"] for ds in datasets for info in ds.samples
                            if info["frame"] not in val_frames)
    first = datasets[0]._read_channels(first_train_path)[0 if args.label == "channel" else 1]
    nr, nt = first.shape[-2:]
    model = SevenSensorFlowModel(
        nr, nt, args.plane_dim, args.latent_resolution, args.render_resolution,
        args.world_extent, args.near, args.far, args.n_samples,
        args.carrier_hz, args.ray_chunk).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) / timestamp
    log_dir = Path("logs") / f"mimo_csi_triplane_nerf2_128_renderdiffusion_flowmatching_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"triplane_flowmatching_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_dir / "logger.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("CSI=%dx%d, train=%d, val=%d, CAVs=1-4, modalities=7, rays=%d, samples=%d",
                nr, nt, len(train_ids), len(val_ids), model.renderer.direction_grid.shape[0], args.n_samples)

    def move(batch):
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    @torch.inference_mode()
    def evaluate(epoch):
        model.eval()
        cos_all, nmse_all = [], []
        save_dir = log_dir / "test" / str(epoch)
        image_dir = save_dir / "pred_csi"
        image_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "all_cos_nmse.txt").open("w", encoding="utf-8") as metrics_file:
            metrics_file.write("index\tframe_id\tcosine_similarity\tnmse_db\n")
            sample_no = 0
            for raw in tqdm(val_loader, desc=f"Test {epoch}", leave=False):
                batch = move(raw)
                target = target_from(batch, args.label)
                pred, weights, _ = model(batch, args.fm_inference_steps)
                cos, nmse = metric(pred, target)
                cos_all.extend(cos.cpu().tolist())
                nmse_all.extend(nmse.cpu().tolist())
                for i in range(pred.shape[0]):
                    frame = str(batch["frame_id"][i])
                    stem = f"{sample_no:05d}_{frame}"
                    metrics_file.write(f"{sample_no}\t{frame}\t{cos[i].item():.8f}\t{nmse[i].item():.8f}\n")
                    pn, tn = pred[i].cpu().numpy(), target[i].cpu().numpy()
                    np.savez_compressed(save_dir / f"{stem}.npz", prediction=pn,
                                        ground_truth=tn, frame_id=frame, label=args.label,
                                        modality_weights=weights[i].cpu().numpy())
                    pm, tm = np.hypot(pn[0], pn[1]), np.hypot(tn[0], tn[1])
                    low = min(float(pm.min()), float(tm.min()))
                    high = max(float(pm.max()), float(tm.max()))
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    img = ax[0].imshow(tm, origin="lower", aspect="auto", cmap="viridis",
                                       vmin=low, vmax=max(high, low + 1e-12))
                    ax[0].set_title("Ground truth CSI |H|")
                    ax[1].imshow(pm, origin="lower", aspect="auto", cmap="viridis",
                                 vmin=low, vmax=max(high, low + 1e-12))
                    ax[1].set_title("Flow-matching CSI |H|")
                    fig.subplots_adjust(bottom=0.2, wspace=0.12)
                    colorbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.035])
                    fig.colorbar(img, cax=colorbar_ax, orientation="horizontal")
                    fig.savefig(image_dir / f"{stem}.png", dpi=150)
                    plt.close(fig)
                    sample_no += 1
        model.train()
        return float(np.mean(cos_all)), float(np.mean(nmse_all))

    history, best_nmse = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        csi_losses, fm_losses, kl_losses, cos_values, nmse_values = [], [], [], [], []
        progress = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
        for raw in progress:
            batch = move(raw)
            target = target_from(batch, args.label)
            prediction, weights, flow_losses = model(batch, args.fm_train_steps, target)
            csi_loss = F.mse_loss(prediction, target)
            fm_image, fm_lidar, kl = flow_losses
            fm_loss = 0.5 * (fm_image + fm_lidar)
            loss = csi_loss + args.fm_loss_weight * fm_loss + args.posterior_kl_weight * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cos, nmse = metric(prediction.detach(), target)
            csi_losses.append(csi_loss.item())
            fm_losses.append(fm_loss.item())
            kl_losses.append(kl.item())
            cos_values.extend(cos.cpu().tolist())
            nmse_values.extend(nmse.cpu().tolist())
            progress.set_postfix(csi_mse=f"{csi_loss.item():.4g}",
                                 fm=f"{fm_loss.item():.4g}",
                                 cosine=f"{cos.mean().item():.4f}",
                                 nmse_db=f"{nmse.mean().item():.2f}")

        record = {"epoch": epoch, "train_loss": float(np.mean(csi_losses)),
                  "train_csi_mse": float(np.mean(csi_losses)),
                  "train_flow_matching_loss": float(np.mean(fm_losses)),
                  "train_posterior_kl": float(np.mean(kl_losses)),
                  "train_cosine": float(np.mean(cos_values)),
                  "train_nmse_db": float(np.mean(nmse_values)),
                  "lr": optimizer.param_groups[0]["lr"]}
        print(json.dumps(record))
        if epoch % args.test_every == 0 or epoch == args.epochs:
            test_cos, test_nmse = evaluate(epoch)
            record.update(test_cosine=test_cos, test_nmse_db=test_nmse)
            print(f"Epoch {epoch}: test cosine={test_cos:.5f}, NMSE={test_nmse:.3f} dB")
            logger.info("Epoch %d test cosine=%.6f NMSE=%.3f dB", epoch, test_cos, test_nmse)
            if test_nmse < best_nmse:
                best_nmse = test_nmse
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "args": vars(args), "nr": nr, "nt": nt, "target_label": args.label},
                           output / "checkpoint_best.pt")
        if epoch >= args.lr_start_epoch and (epoch - args.lr_start_epoch) % args.lr_step_size == 0:
            scheduler.step()
            record["lr"] = optimizer.param_groups[0]["lr"]
        history.append(record)
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "args": vars(args), "nr": nr, "nt": nt, "target_label": args.label}
            torch.save(state, output / f"checkpoint_{epoch}.pt")
            torch.save(state, output / "checkpoint_latest.pt")

    handler.close()
    logger.removeHandler(handler)


if __name__ == "__main__":
    main()
