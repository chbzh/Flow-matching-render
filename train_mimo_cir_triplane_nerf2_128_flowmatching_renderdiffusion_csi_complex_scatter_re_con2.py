"""CSI-endpoint flow model with a multimodal generative initializer.

Camera and LiDAR feature heads belong to one shared multimodal initializer,
which samples a CSI-shaped x0. During training one random t is sampled and a
single interpolated CSI state is rendered back to the CSI endpoint. Inference
iteratively updates CSI states using the rendered endpoint estimate.
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


class GeneratorMatchedCAVCSIDataset(MultiSensorCAVCSIDataset):
    """Match radio rotations in the published MMW CIR generator.

    Only RADIO orientations change. LiDAR point transforms retain CARLA's
    actual rotation. NPZ local/global path angles verify this convention.
    """
    @staticmethod
    def _pose_orientation(pose):
        rot = pose["rotation"]
        yaw, pitch = -math.radians(rot["yaw"]), math.radians(rot["pitch"])
        # The generator reads roll from location, not rotation (normally 0).
        # Reproduce the dataset generator rather than fixing it retroactively.
        roll = math.radians(pose["location"].get("roll", 0.0))
        cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw),
                                math.cos(pitch), math.sin(pitch),
                                math.cos(roll), math.sin(roll))
        matrix = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                           [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                           [-sp, cp*sr, cp*cr]], dtype=np.float32)
        return matrix.reshape(-1)


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


class MultimodalCSIInitializer(nn.Module):
    """One generative initializer with camera and LiDAR feature heads."""
    def __init__(self, nr, nt, feature_dim=256, world_extent=100.0):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.world_extent = world_extent
        # Unit expected TOTAL noise energy, rather than variance 1 in each
        # of 2*Nr*Nt entries (which overwhelms max-normalized sparse CSI).
        self.csi_entry_scale = 1.0 / math.sqrt(2 * nr * nt)
        self.camera_feature_head = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, feature_dim), nn.GELU())
        self.camera_slot_embedding = nn.Parameter(torch.randn(5, feature_dim) * 0.02)
        self.lidar_pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.GELU())
        self.lidar_feature_head = nn.Sequential(
            nn.Linear(256, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.lidar_slot_embedding = nn.Parameter(torch.randn(2, feature_dim) * 0.02)
        self.geometry_feature_head = nn.Sequential(
            nn.Linear(24, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(3 * feature_dim, 2 * feature_dim), nn.LayerNorm(2 * feature_dim),
            nn.GELU(), nn.Linear(2 * feature_dim, feature_dim), nn.LayerNorm(feature_dim),
            nn.GELU())
        csi_size = 2 * nr * nt
        self.to_mean = nn.Linear(feature_dim, csi_size)
        self.to_logvar = nn.Linear(feature_dim, csi_size)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.zeros_(self.to_logvar.bias)

    def encode_points(self, points, origin):
        normalized = ((points - origin[:, None]) / self.world_extent).clamp(-1, 1)
        features = self.lidar_pointnet(normalized.transpose(1, 2).contiguous()).amax(dim=-1)
        return self.lidar_feature_head(features)

    def forward(self, batch, sample=True):
        batch_size = batch["cav_images"].shape[0]
        images = torch.cat((batch["cav_images"], batch["rsu_image"].unsqueeze(1)), dim=1)
        image_features = self.camera_feature_head(
            images.reshape(-1, *images.shape[2:])).reshape(batch_size, 5, -1)
        image_features = image_features + self.camera_slot_embedding.unsqueeze(0)
        camera_summary = image_features.mean(dim=1)

        lidar_features = torch.stack((self.encode_points(batch["cav_points"], batch["tx_pos"]),
                                      self.encode_points(batch["points"], batch["tx_pos"])), dim=1)
        lidar_features = lidar_features + self.lidar_slot_embedding.unsqueeze(0)
        lidar_summary = lidar_features.mean(dim=1)
        geometry = torch.cat(((batch["rx_pos"] - batch["tx_pos"]) / self.world_extent,
                              batch["tx_pos"] / self.world_extent,
                              batch["rx_orientation"].reshape(batch_size, 9),
                              batch["tx_orientation"].reshape(batch_size, 9)), dim=-1)
        geometry_summary = self.geometry_feature_head(geometry)
        fused = self.fusion(torch.cat((camera_summary, lidar_summary, geometry_summary), dim=-1))

        mean = self.to_mean(fused).reshape(batch_size, 2, self.nr, self.nt) * self.csi_entry_scale
        logvar = self.to_logvar(fused).reshape_as(mean).clamp(-8, 2)
        if sample:
            x0 = mean + self.csi_entry_scale * torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            x0 = mean
        return x0, mean, logvar, fused


class FlowUNetResidualBlock(nn.Module):
    """Residual Conv2D block modulated by the flow-time embedding."""
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition_projection = nn.Linear(condition_dim, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, 1)

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition_projection(F.silu(condition))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class CSIToTriPlaneUNet(nn.Module):
    """Single time-conditioned U-Net mapping CSI state xt to a tri-plane."""
    def __init__(self, plane_dim, resolution=32, condition_dim=256,
                 base_width=64, depth=3):
        super().__init__()
        if depth < 2:
            raise ValueError("Flow U-Net depth must be at least 2")
        self.plane_dim = plane_dim
        self.resolution = resolution
        output_channels = 3 * plane_dim
        self.time_projection = nn.Sequential(
            nn.Linear(128, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.widths = [base_width * (2 ** level) for level in range(depth + 1)]
        self.input = nn.Conv2d(2, self.widths[0], 3, padding=1)
        self.encoders = nn.ModuleList([
            FlowUNetResidualBlock(self.widths[level], self.widths[level], condition_dim)
            for level in range(depth + 1)])
        self.downsamples = nn.ModuleList([
            nn.Conv2d(self.widths[level], self.widths[level + 1], 3,
                      stride=2, padding=1) for level in range(depth)])
        self.bottleneck = nn.ModuleList([
            FlowUNetResidualBlock(self.widths[-1], self.widths[-1], condition_dim),
            FlowUNetResidualBlock(self.widths[-1], self.widths[-1], condition_dim)])
        self.upsamples = nn.ModuleList([
            nn.Conv2d(self.widths[level + 1], self.widths[level], 3, padding=1)
            for level in range(depth)])
        self.decoders = nn.ModuleList([
            FlowUNetResidualBlock(2 * self.widths[level], self.widths[level], condition_dim)
            for level in range(depth)])
        self.output = nn.Sequential(
            nn.GroupNorm(group_count(self.widths[0]), self.widths[0]), nn.SiLU(),
            nn.Conv2d(self.widths[0], output_channels, 3, padding=1))

    def forward(self, x_t, t):
        cond = self.time_projection(sinusoidal_time(t, 128))
        x = F.interpolate(x_t, size=(self.resolution, self.resolution),
                           mode="bilinear", align_corners=False)
        x = self.input(x)
        skips = []
        for level, block in enumerate(self.encoders):
            x = block(x, cond)
            if level < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[level](x)
        for block in self.bottleneck:
            x = block(x, cond)
        for level in reversed(range(len(self.decoders))):
            skip = skips[level]
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = self.upsamples[level](x)
            x = torch.cat((x, skip), dim=1)
            x = self.decoders[level](x, cond)
        planes = self.output(x).reshape(
            x_t.shape[0], 3, self.plane_dim, self.resolution, self.resolution)
        return planes

class ComplexScatterCSIField(nn.Module):
    """Tri-plane field producing a complex coefficient and density.

    In received mode the coefficient includes end-to-end gain and phase.
    In scatter mode it is local rho, excluding free-space propagation.
    """
    def __init__(self, nr, nt, plane_dim=64, condition_dim=256, hidden_dim=256,
                 world_extent=100.0, position_frequencies=6, direction_frequencies=6,
                 initial_density=0.01):
        super().__init__()
        self.world_extent = world_extent
        self.position_frequencies = position_frequencies
        self.direction_frequencies = direction_frequencies
        self.rx_position = SinusoidalPositionEncoding(128)
        pos_dim = 3 * (1 + 2 * position_frequencies)
        view_dim = 3 * (1 + 2 * direction_frequencies)
        in_dim = 3 * plane_dim + pos_dim + view_dim + 2 * condition_dim + 2 * 128
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.skip = nn.Linear(in_dim, hidden_dim)
        # The renderer's coefficient_mode defines the coefficient's semantics.
        self.output = nn.Linear(hidden_dim, 3)
        # Start with order-one optical thickness over the marching interval,
        # rather than softplus(0)~0.69/m, which hides almost all far samples.
        if initial_density <= 0:
            raise ValueError("initial_density must be positive")
        with torch.no_grad():
            self.output.weight[2].zero_()
            self.output.bias[2].fill_(math.log(math.expm1(initial_density)))

    def forward(self, planes, points, directions, tx_pos, fused, rx_pos, tx_pe, condition):
        batch, views, samples, _ = points.shape
        local = ((points - tx_pos[:, None, None, :]) / self.world_extent).clamp(-1, 1)
        plane_features = NeRF2CSIField.sample_planes(planes, local)
        pos = NeRF2CSIField.positional_encode(local, self.position_frequencies)
        view = NeRF2CSIField.positional_encode(
            directions[:, :, None].expand(-1, -1, samples, -1), self.direction_frequencies)
        rx_pe = self.rx_position(rx_pos / 100.0)
        contexts = [
            fused[:, None, None].expand(-1, views, samples, -1),
            rx_pe[:, None, None].expand(-1, views, samples, -1),
            tx_pe[:, None, None].expand(-1, views, samples, -1),
            condition[:, None, None].expand(-1, views, samples, -1)]
        inputs = torch.cat((plane_features, pos, view, *contexts), dim=-1)
        hidden = self.trunk(inputs) + self.skip(inputs)
        return self.output(F.relu(hidden))


class RxOriginCSIRenderer(nn.Module):
    """RX-origin ray marcher with coherent TX/RX array steering responses.

    Sionna 1xN single-polarization ULAs lie on local +Y. Orientations are
    local-to-world rotation matrices, as returned by MIMOCSIDataset.
    Both bearings point FROM the array center TOWARD the scatterer.
    """
    def __init__(self, nr, nt, near=0.0, far=100.0, n_samples=32,
                 carrier_hz=28e9, ray_chunk=4, elevation_count=9,
                 azimuth_count=36, latent_resolution=16,
                 array_spacing_wavelength=0.5, coefficient_mode="received"):
        super().__init__()
        if (nr < 1 or nt < 1 or near < 0 or far <= near or n_samples < 2
                or carrier_hz <= 0 or ray_chunk < 1 or elevation_count < 1
                or azimuth_count < 1 or array_spacing_wavelength <= 0):
            raise ValueError("Invalid CSI renderer geometry or sampling settings")
        self.nr, self.nt = nr, nt
        self.near, self.far, self.n_samples = near, far, n_samples
        self.carrier_hz, self.ray_chunk = carrier_hz, ray_chunk
        self.latent_resolution = latent_resolution
        self.array_spacing_wavelength = array_spacing_wavelength
        if coefficient_mode not in ("received", "scatter"):
            raise ValueError("coefficient_mode must be received or scatter")
        self.coefficient_mode = coefficient_mode
        # Equal-solid-angle cell centers on the WHOLE sphere. The old grid
        # excluded the horizon, lower hemisphere, and all elevations below 10deg.
        mu = -1.0 + (torch.arange(elevation_count) + 0.5) * (2.0 / elevation_count)
        azimuth = torch.arange(azimuth_count) * (2.0 * math.pi / azimuth_count)
        z, az = torch.meshgrid(mu, azimuth, indexing="ij")
        radius = torch.sqrt((1.0 - z.square()).clamp_min(0))
        dirs = torch.stack((radius * az.cos(), radius * az.sin(), z), dim=-1)
        self.register_buffer("direction_grid", dirs.reshape(-1, 3))
        self.register_buffer("angle_weights", torch.full(
            (elevation_count * azimuth_count,),
            4.0 * math.pi / (elevation_count * azimuth_count)))

    def _steering(self, direction, element_count, wavelength, orientation):
        # direction [B,...,3]; orientation [B,9] or [B,3,3], local -> world.
        # For e^{-jkL}, moving an element TOWARD the scatterer shortens the
        # path, hence exp(+jk u.dot(offset)) at BOTH endpoints. Equivalently
        # H=beta*a_rx*a_tx^H when a_tx uses the opposite phase convention.
        rotation = orientation.to(direction).reshape(direction.shape[0], 3, 3)
        axis = rotation[:, :, 1]  # second COLUMN, not second row
        axis = axis.reshape(direction.shape[0], *([1] * (direction.ndim - 2)), 3)
        projection = (direction * axis).sum(dim=-1, keepdim=True)
        index = torch.arange(element_count, device=direction.device, dtype=direction.dtype)
        offsets_wavelengths = (index - (element_count - 1) / 2.0) * self.array_spacing_wavelength
        phase = 2.0 * math.pi * projection * offsets_wavelengths
        return torch.exp(1j * phase)

    def render(self, field, planes, tx_pos, rx_pos, fused, tx_pe, condition,
               rx_orientation, tx_orientation):
        batch = rx_pos.shape[0]
        # Triplane features are generated at lower latent resolution for memory
        # efficiency, then resized before spatial queries.
        planes = F.interpolate(
            planes.reshape(batch * 3, planes.shape[2], *planes.shape[-2:]),
            size=(self.latent_resolution, self.latent_resolution), mode="bilinear",
            align_corners=False).reshape(batch, 3, planes.shape[2],
                                         self.latent_resolution, self.latent_resolution)
        directions_all = self.direction_grid.to(rx_pos).unsqueeze(0).expand(batch, -1, -1)
        # Midpoint quadrature excludes RX itself even when near=0, and the
        # segment widths integrate exactly to far-near (no extra last bin).
        edges = torch.linspace(self.near, self.far, self.n_samples + 1,
                               device=rx_pos.device, dtype=rx_pos.dtype)
        t_vals = 0.5 * (edges[:-1] + edges[1:])
        delta = (edges[1:] - edges[:-1]).view(1, 1, -1)
        wavelength = 299792458.0 / self.carrier_hz
        accumulated = torch.zeros(batch, self.nr, self.nt, device=rx_pos.device,
                                  dtype=torch.complex64 if rx_pos.dtype == torch.float32 else torch.complex128)
        for start in range(0, directions_all.shape[1], self.ray_chunk):
            directions = directions_all[:, start:start + self.ray_chunk]
            views = directions.shape[1]
            depths = t_vals.view(1, 1, -1).expand(batch, views, -1)
            # RX anchors rendering rays. TX remains an explicit field condition.
            points = rx_pos[:, None, None, :] + directions[:, :, None, :] * depths[..., None]
            raw = field(planes, points, directions, tx_pos, fused, rx_pos, tx_pe, condition)
            raw = raw.reshape(batch, views, self.n_samples, 3)
            rho = torch.complex(raw[..., 0], raw[..., 1])
            sigma = F.softplus(raw[..., 2])
            optical_depth = sigma * delta
            alpha = -torch.expm1(-optical_depth)
            prefix = torch.cat((torch.zeros_like(optical_depth[:, :, :1]),
                                optical_depth[:, :, :-1]), dim=2)
            transmittance = torch.exp(-torch.cumsum(prefix, dim=2))
            weights = alpha * transmittance

            rx_bearing = directions
            tx_bearing = points - tx_pos[:, None, None, :]
            tx_bearing = F.normalize(tx_bearing, dim=-1, eps=1e-8)
            a_rx = self._steering(rx_bearing, self.nr, wavelength, rx_orientation)
            a_tx = self._steering(tx_bearing, self.nt, wavelength, tx_orientation)

            # Optional explicit bistatic free-space propagation. Electric-field amplitude
            # decays as 1/d on each leg, while phase depends on total path length.
            # Clamp only protects the singularity if a sample coincides with an
            # antenna; distances and coordinates are assumed to be in meters.
            if self.coefficient_mode == "scatter":
                distance_tx = torch.linalg.vector_norm(
                    points - tx_pos[:, None, None, :], dim=-1).clamp_min(1e-3)
                distance_rx = torch.linalg.vector_norm(
                    points - rx_pos[:, None, None, :], dim=-1).clamp_min(1e-3)
                path_length = distance_tx + distance_rx
                phase = torch.remainder(
                    path_length.double() * (-2.0 * math.pi / wavelength), 2.0 * math.pi)
                propagation = torch.exp(1j * phase).to(accumulated.dtype)
                propagation = propagation / (distance_tx * distance_rx)
                rho = rho * propagation
            # received mode learns beta=rho*propagation directly. It still
            # renders via the RX/TX manifolds and complex coherent summation;
            # it does not discard phase or directly output a CSI matrix.

            # [B,V,S,Nr,Nt], positive phase for both outward bearings.
            point_csi = (rho[..., None, None] * a_rx[:, :, None, :, None]
                         * a_tx[:, :, :, None, :])
            ray_csi = torch.sum(weights[..., None, None] * point_csi, dim=2)
            angle_weights = self.angle_weights[start:start + views].to(ray_csi)
            accumulated = accumulated + torch.sum(
                ray_csi * angle_weights[None, :, None, None], dim=1)
        return accumulated


class CSISpaceFlowModel(nn.Module):
    """One CSI-state flow, with rendering embedded in endpoint prediction."""
    def __init__(self, nr, nt, plane_dim=64, plane_resolution=32,
                 render_resolution=32, world_extent=100.0, near=0.0,
                 far=100.0, n_samples=32, carrier_hz=28e9, ray_chunk=4,
                 array_spacing_wavelength=0.5,
                 condition_dim=256, flow_unet_depth=3, flow_base_width=64,
                 target_label="channel", elevation_count=9, azimuth_count=36,
                 t_zero_probability=0.25, coefficient_mode="received",
                 use_learned_norm=False, normalize_csi=False):
        super().__init__()
        self.nr, self.nt = nr, nt
        self.plane_dim, self.plane_resolution = plane_dim, plane_resolution
        self.condition_dim = condition_dim
        if target_label not in ("channel", "channel_H"):
            raise ValueError("target_label must be channel or channel_H")
        self.target_label = target_label
        self.use_learned_norm = use_learned_norm
        self.normalize_csi = normalize_csi
        if use_learned_norm and normalize_csi:
            raise ValueError('learned-norm and normalize-csi are mutually exclusive')
        if not 0 <= t_zero_probability <= 1:
            raise ValueError("t_zero_probability must be in [0,1]")
        self.t_zero_probability = t_zero_probability
        self.initializer = MultimodalCSIInitializer(nr, nt, condition_dim, world_extent)
        self.flow = CSIToTriPlaneUNet(
            plane_dim, plane_resolution, condition_dim, flow_base_width, flow_unet_depth)
        self.tx_position = SinusoidalPositionEncoding(128)
        self.field = ComplexScatterCSIField(
            nr, nt, plane_dim=plane_dim, condition_dim=condition_dim,
            world_extent=world_extent, initial_density=1.0 / (far - near))
        self.renderer = RxOriginCSIRenderer(
            nr, nt, near, far, n_samples, carrier_hz, ray_chunk,
            latent_resolution=render_resolution,
            elevation_count=elevation_count, azimuth_count=azimuth_count,
            coefficient_mode=coefficient_mode,
            array_spacing_wavelength=array_spacing_wavelength)
        # Keep the head for state-dict compatibility, but bypass it by default.
        # Only use_learned_norm enables L2 direction plus a predicted magnitude.
        self.channel_norm_head = nn.Sequential(
            nn.Linear(3 * plane_dim, condition_dim), nn.SiLU(),
            nn.Linear(condition_dim, 1))
        nn.init.zeros_(self.channel_norm_head[-1].weight)
        nn.init.constant_(self.channel_norm_head[-1].bias, math.log(math.expm1(1.0)))

    @staticmethod
    def spatial_to_target(complex_csi, label):
        if label == "channel":
            # Dataset: F_rx^H @ H @ F_tx, F[n,k]=exp(-j2pi*n*k/N)/sqrt(N).
            # This is IFFT on RX and FFT on TX, with NO fftshift.
            return torch.fft.fft(torch.fft.ifft(complex_csi, dim=-2, norm="ortho"),
                                 dim=-1, norm="ortho")
        if label != "channel_H":
            raise ValueError("label must be channel or channel_H")
        return complex_csi

    def render_endpoint(self, planes, batch):
        # The flow state carries the sensor/CSI context. The NeRF field receives
        # only its spatial planes plus the TX/RX geometry needed for rendering.
        batch_size = planes.shape[0]
        fused = planes.new_zeros(batch_size, self.condition_dim)
        condition = torch.zeros_like(fused)
        tx_pe = self.tx_position(batch["tx_pos"] / 100.0)
        complex_csi = self.renderer.render(
            self.field, planes, batch["tx_pos"], batch["rx_pos"], fused, tx_pe, condition,
            batch["rx_orientation"], batch["tx_orientation"])
        complex_csi = self.spatial_to_target(complex_csi, self.target_label)
        if self.use_learned_norm:
            raw_norm = torch.linalg.vector_norm(complex_csi.flatten(1), dim=1).view(-1, 1, 1)
            pooled = planes.mean(dim=(-2, -1)).flatten(1)
            predicted_norm = F.softplus(self.channel_norm_head(pooled)).view(-1, 1, 1)
            complex_csi = complex_csi / raw_norm.clamp_min(1e-8) * predicted_norm
        if self.normalize_csi:
            peak = complex_csi.abs().amax(dim=(-2, -1), keepdim=True)
            complex_csi = complex_csi / peak.clamp_min(1e-8)
        return torch.stack((complex_csi.real, complex_csi.imag), dim=1)

    def predict_endpoint(self, x_t, t, batch):
        planes = self.flow(x_t, t)
        return self.render_endpoint(planes, batch), planes

    def training_prediction(self, batch, target, return_initializer=False):
        """One random-t interpolation and one rendered endpoint prediction."""
        x0, mean, _, _ = self.initializer(batch, sample=True)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t = t.clamp(max=1.0 - 1e-4)
        t = torch.where(torch.rand_like(t) < self.t_zero_probability, torch.zeros_like(t), t)
        view_t = t[:, None, None, None]
        x_t = (1.0 - view_t) * x0 + view_t * target
        endpoint, planes = self.predict_endpoint(x_t, t, batch)
        result = (endpoint, planes, x0, t)
        return result + (mean,) if return_initializer else result

    @torch.inference_mode()
    def sample(self, batch, steps=8, reparameterize=True):
        """Iterate CSI states from the multimodal posterior sample to t=1."""
        if steps < 1:
            raise ValueError("Inference steps must be positive")
        x_t, _, _, _ = self.initializer(batch, sample=reparameterize)
        batch_size = x_t.shape[0]
        final_planes = None
        for step in range(steps):
            t_value = step / steps
            t_next_value = (step + 1) / steps
            t = torch.full((batch_size,), t_value, device=x_t.device, dtype=x_t.dtype)
            x1_at_t, final_planes = self.predict_endpoint(x_t, t, batch)
            # Re-estimate x0|t from the CURRENT state and endpoint prediction at
            # every step; do not carry forward the initial or previous x0 estimate.
            denom = max(1.0 - t_value, 1e-6)
            x0_at_t = (x_t - t_value * x1_at_t) / denom
            x_t = (1.0 - t_next_value) * x0_at_t + t_next_value * x1_at_t
        return x_t, final_planes


def target_from(batch, label, normalize=False):
    target = batch[label].float()
    if not normalize:
        return target
    scale = torch.linalg.vector_norm(target, dim=1).amax(dim=(-2, -1), keepdim=True).unsqueeze(1)
    return target / scale.clamp_min(1e-8)


def metric(pred, target):
    p, t = pred.flatten(1), target.flatten(1)
    cosine = F.cosine_similarity(p, t, dim=1)
    nmse = (p - t).square().sum(1) / t.square().sum(1).clamp_min(1e-12)
    return cosine, 10 * torch.log10(nmse.clamp_min(1e-12))


class CrossModalContrastiveLoss(nn.Module):
    """One-way InfoNCE: sampled encoder x0 against paired ground-truth CSI."""
    def __init__(self, initial_temperature=0.07):
        super().__init__()
        if not 1e-3 <= initial_temperature <= 1.0:
            raise ValueError('contrastive temperature must be in [0.001, 1]')
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))

    @property
    def temperature(self):
        return self.log_temperature.clamp(math.log(1e-3), 0.).exp()

    def forward(self, x0, target):
        if x0.shape != target.shape or x0.ndim < 2 or x0.shape[0] < 1:
            raise ValueError('Expected matching nonempty batched x0 and target CSI')
        if x0.shape[0] == 1:
            return x0.sum() * 0.0  # No negative samples in a singleton batch.
        source = F.normalize(x0.flatten(1).float(), dim=1, eps=1e-8)
        truth = F.normalize(target.detach().flatten(1).float(), dim=1, eps=1e-8)
        logits = (source @ truth.T) / self.temperature
        return F.cross_entropy(logits, torch.arange(x0.shape[0], device=x0.device))


def csi_nmse_loss(pred, target):
    """Equal sample weighting, aligned with the reported linear NMSE."""
    error = (pred - target).flatten(1).square().sum(1)
    energy = target.flatten(1).square().sum(1).clamp_min(1e-12)
    return (error / energy).mean()


def make_loaders(args, device):
    cav_ids = (1, 2, 3, 4) if args.use_all_cavs else (2,)
    cav_datasets = [GeneratorMatchedCAVCSIDataset(
        args.sensor_root, args.channel_root, args.scenario, args.channel_scenario,
        cav_id=cav_id, cav_num_points=args.cav_num_points,
        num_points=args.num_points, frequency_offset_hz=args.frequency_offset_hz,
        seed=args.seed, rx_pose_key=args.rx_pose_key, tx_pose_key=args.tx_pose_key,
        image_size=args.image_size) for cav_id in cav_ids]
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
    if args.overfit_samples:
        train_ids = train_ids[:args.overfit_samples]
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True, **common)
    val_loader = DataLoader(Subset(dataset, val_ids), shuffle=False, **common)
    return cav_datasets, train_loader, val_loader, train_ids, val_ids, val_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-root", default="/root/autodl-tmp/Document/cbz/CSI_sensor_data")
    parser.add_argument("--channel-root", default="/root/autodl-tmp/Document/cbz/CSI_data")
    parser.add_argument("--scenario", default="Town03_Tjunction_wiz_slope_seed42")
    parser.add_argument("--channel-scenario", default="Town03_Tjunction")
    # Published Multimodal-Wireless v1.1 generation uses lidar_pose for radio
    # endpoints too (Sionna_channel_storage/data_utils.py: load_loc_speed).
    parser.add_argument("--rx-pose-key", choices=("vehicle_pose", "lidar_pose"), default="lidar_pose")
    parser.add_argument("--tx-pose-key", choices=("rsu_pose", "lidar_pose"), default="lidar_pose")
    parser.add_argument("--label", choices=("channel", "channel_H"), default="channel")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--cav-num-points", type=int, default=24000)
    parser.add_argument("--use-all-cavs", action="store_true",
                        help="Use datasets from CAV1-CAV4; default uses only CAV1")
    parser.add_argument("--plane-dim", type=int, default=64,
                        help="Feature channels per generated plane")
    parser.add_argument("--learned-norm", dest="use_learned_norm",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Use L2-normalized CSI times learned norm; default returns raw rendered CSI")
    parser.add_argument("--normalize-csi", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize BOTH rendered and target CSI by their own peak complex magnitude; default keeps both raw")
    parser.add_argument("--plane-resolution", type=int, default=32,
                        help="Spatial size of each generated feature plane")
    parser.add_argument("--render-resolution", type=int, default=32,
                        help="Feature-plane size used by the CSI renderer")
    parser.add_argument("--world-extent", type=float, default=100.0)
    parser.add_argument("--near", type=float, default=0.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--carrier-hz", type=float, default=28e9,
                        help="Must match the CIR generator; historical MMW data uses 28 GHz")
    parser.add_argument("--coefficient-mode", choices=("received", "scatter"), default="received",
                        help="Learn full received coefficients, or explicitly multiply local rho by propagation")
    parser.add_argument("--ray-chunk", type=int, default=16)
    parser.add_argument("--elevation-count", type=int, default=9)
    parser.add_argument("--azimuth-count", type=int, default=36)
    parser.add_argument("--array-spacing-wavelength", type=float, default=0.5,
                        help="1xN ULA spacing in wavelengths along each array's rotated local Y")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive-loss", dest="use_contrastive_loss",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable x0-to-CSI contrastive loss; disabled by default")
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07,
                        help="Initial trainable temperature in [0.001, 1]")
    parser.add_argument("--t-zero-probability", type=float, default=0.25,
                        help="Fraction of training examples with no target mixed into xt")
    parser.add_argument("--overfit-samples", type=int, default=0,
                        help="Fit only the first N training samples; evaluate those samples without plots")
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-start-epoch", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--inference-steps", "--flow-inference-steps", dest="inference_steps",
                        type=int, default=1,
                        help="Number T of CSI-state updates during inference")
    parser.add_argument("--test-reparameterize", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Reparameterize the test initializer (default); --no-test-reparameterize uses encoder mu")
    parser.add_argument("--flow-unet-depth", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--flow-base-width", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency-offset-hz", type=float, default=0.0)
    parser.add_argument("--output", default="/root/autodl-tmp/Document/cbz/WRF-GSplus-main-4angents/runs/mimo_csi_triplane_nerf2_128_renderdiffusion_flowmatching_csi_complex_scatter_re_con")
    args, unknown = parser.parse_known_args()
    if args.use_learned_norm and args.normalize_csi:
        parser.error('--learned-norm cannot be combined with --normalize-csi')
    if (args.near < 0 or args.far <= args.near or args.n_samples < 2
            or args.render_resolution < 4 or args.flow_base_width < 8
            or args.plane_resolution < 2 ** args.flow_unet_depth
            or args.inference_steps < 1 or args.array_spacing_wavelength <= 0
            or args.carrier_hz <= 0 or args.ray_chunk < 1
            or args.elevation_count < 1 or args.azimuth_count < 1
            or args.overfit_samples < 0
            or args.contrastive_loss_weight < 0
            or not 1e-3 <= args.contrastive_temperature <= 1.0
            or not 0 <= args.t_zero_probability <= 1):
        raise ValueError("Invalid ray range, plane resolution, sample count, or inference step count")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, train_loader, val_loader, train_ids, val_ids, val_frames = make_loaders(args, device)
    first_train_path = next(info["cir"] for ds in datasets for info in ds.samples
                            if info["frame"] not in val_frames)
    first = datasets[0]._read_channels(first_train_path)[0 if args.label == "channel" else 1]
    nr, nt = first.shape[-2:]
    model = CSISpaceFlowModel(
        nr, nt, args.plane_dim, args.plane_resolution, args.render_resolution,
        args.world_extent, args.near, args.far, args.n_samples,
        args.carrier_hz, args.ray_chunk,
        array_spacing_wavelength=args.array_spacing_wavelength,
        target_label=args.label, elevation_count=args.elevation_count,
        azimuth_count=args.azimuth_count,
        flow_unet_depth=args.flow_unet_depth,
        flow_base_width=args.flow_base_width,
        coefficient_mode=args.coefficient_mode,
        t_zero_probability=args.t_zero_probability,
        use_learned_norm=args.use_learned_norm,
        normalize_csi=args.normalize_csi).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    contrastive_criterion = None
    train_parameters = list(model.parameters())
    if args.use_contrastive_loss and args.contrastive_loss_weight > 0:
        contrastive_criterion = CrossModalContrastiveLoss(args.contrastive_temperature).to(device)
        optimizer.add_param_group({"params": list(contrastive_criterion.parameters()), "weight_decay": 0.0})
        train_parameters += list(contrastive_criterion.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) / timestamp
    log_dir = Path("/root/autodl-tmp/Document/cbz/WRF-GSplus-main-4angents/logs") / f"mimo_csi_triplane_nerf2_128_renderdiffusion_flowmatching_csi_complex_scatter_re_con_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"csi_space_flowmatching_complex_scatter_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_dir / "logger.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    selected_cavs = "1-4" if args.use_all_cavs else "1"
    logger.info("CSI=%dx%d, train=%d, val=%d, CAVs=%s, one shared CSI flow, rays=%d, samples=%d",
                nr, nt, len(train_ids), len(val_ids), selected_cavs,
                model.renderer.direction_grid.shape[0], args.n_samples)
    geometry_summary = (f"label={args.label}, carrier={args.carrier_hz:g} Hz, "
                        f"ULA=local Y, spacing={args.array_spacing_wavelength:g} wavelengths, "
                        f"RX pose={args.rx_pose_key}, TX pose={args.tx_pose_key}, "
                        f"full-sphere grid={args.elevation_count}x{args.azimuth_count}, "
                        f"coefficient_mode={args.coefficient_mode}, "
                        f"output={'peak-normalized-CSI' if args.normalize_csi else 'L2-direction+learned-norm' if args.use_learned_norm else 'raw-rendered-CSI'}, "
                        f"target={'peak-normalized-CSI' if args.normalize_csi else 'raw-CSI'}")
    print(geometry_summary)
    logger.info(geometry_summary)

    def move(batch):
        return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    @torch.inference_mode()
    def evaluate(epoch):
        model.eval()
        if args.overfit_samples:
            # The purpose is training-set fitting, not generalization. Use the
            # initializer mean to separate reconstruction from sampling noise.
            result = {}
            for mode in ("initializer", "t0", "sample"):
                cosines, nmses = [], []
                for raw in train_loader:
                    batch = move(raw)
                    target = target_from(batch, args.label, normalize=args.normalize_csi)
                    mean = model.initializer(batch, sample=False)[0]
                    if mode == "initializer":
                        pred = mean
                    elif mode == "t0":
                        pred = model.predict_endpoint(mean, mean.new_zeros(mean.shape[0]), batch)[0]
                    else:
                        pred = model.sample(batch, args.inference_steps,
                                            reparameterize=args.test_reparameterize)[0]
                    cos, nmse = metric(pred, target)
                    cosines.extend(cos.cpu().tolist())
                    nmses.extend(nmse.cpu().tolist())
                result[mode] = dict(cosine=float(np.mean(cosines)), nmse_db=float(np.mean(nmses)))
            print("[Overfit diagnostic] " + json.dumps(result))
            logger.info("Overfit diagnostic %s", json.dumps(result))
            model.train()
            return result["sample"]["cosine"], result["sample"]["nmse_db"]
        cos_all, nmse_all = [], []
        save_dir = log_dir / "test" / str(epoch)
        image_dir = save_dir / "pred_csi"
        image_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "all_cos_nmse.txt").open("w", encoding="utf-8") as metrics_file:
            metrics_file.write("index\tframe_id\tcosine_similarity\tnmse_db\n")
            sample_no = 0
            for raw in tqdm(val_loader, desc=f"Test {epoch}", leave=False):
                batch = move(raw)
                target = target_from(batch, args.label, normalize=args.normalize_csi)
                pred, planes = model.sample(batch, args.inference_steps,
                                            reparameterize=args.test_reparameterize)
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
                                        triplane=planes[i].cpu().numpy())
                    pm, tm = np.hypot(pn[0], pn[1]), np.hypot(tn[0], tn[1])
                    low = min(float(pm.min()), float(tm.min()))
                    high = max(float(pm.max()), float(tm.max()))
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    img = ax[0].imshow(tm, origin="lower", aspect="auto", cmap="viridis",
                                       vmin=low, vmax=max(high, low + 1e-12))
                    ax[0].set_title("Ground truth CSI |H|")
                    ax[1].imshow(pm, origin="lower", aspect="auto", cmap="viridis",
                                 vmin=low, vmax=max(high, low + 1e-12))
                    ax[1].set_title("Iterative flow-rendered CSI |H|")
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
        csi_losses, cos_values, nmse_values, sampled_t_values = [], [], [], []
        total_losses, energy_ratios, t0_nmses, t0_cosines = [], [], [], []
        contrastive_losses = []
        progress = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
        for raw in progress:
            batch = move(raw)
            target = target_from(batch, args.label, normalize=args.normalize_csi)
            prediction, planes, x0, sampled_t = model.training_prediction(batch, target)
            per_sample_mse = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(dim=1)
            csi_loss = (per_sample_mse / (1.0 + sampled_t).square()).mean()
            contrastive_loss = (contrastive_criterion(x0, target) if contrastive_criterion is not None
                                else x0.new_zeros(()))
            loss = csi_loss + args.contrastive_loss_weight * contrastive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 1.0)
            optimizer.step()
            cos, nmse = metric(prediction.detach(), target)
            csi_losses.append(F.mse_loss(prediction.detach(), target).item())
            total_losses.append(loss.item())
            contrastive_losses.append(contrastive_loss.item())
            energy_ratios.extend((x0.detach().flatten(1).square().sum(1) /
                                  target.flatten(1).square().sum(1).clamp_min(1e-12)).cpu().tolist())
            mask = sampled_t == 0
            t0_nmses.extend(nmse[mask].cpu().tolist())
            t0_cosines.extend(cos[mask].cpu().tolist())
            sampled_t_values.extend(sampled_t.detach().cpu().tolist())
            cos_values.extend(cos.cpu().tolist())
            nmse_values.extend(nmse.cpu().tolist())
            progress.set_postfix(weighted_mse=f"{csi_loss.item():.4g}",
                                 t=f"{sampled_t.mean().item():.3f}",
                                 cosine=f"{cos.mean().item():.4f}",
                                 nmse_db=f"{nmse.mean().item():.2f}")

        record = {"epoch": epoch, "train_loss": float(np.mean(total_losses)),
                  "train_csi_mse": float(np.mean(csi_losses)),
                  "train_contrastive_loss": float(np.mean(contrastive_losses)),
                  "contrastive_temperature": float(contrastive_criterion.temperature.item()) if contrastive_criterion is not None else None,
                  "train_mean_sampled_t": float(np.mean(sampled_t_values)),
                  "train_cosine": float(np.mean(cos_values)),
                  "train_nmse_db": float(np.mean(nmse_values)),
                  "train_x0_target_energy_ratio": float(np.mean(energy_ratios)),
                  "train_t0_nmse_db": float(np.mean(t0_nmses)) if t0_nmses else None,
                  "train_t0_cosine": float(np.mean(t0_cosines)) if t0_cosines else None,
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
                            "contrastive_loss": contrastive_criterion.state_dict() if contrastive_criterion is not None else None,
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
                     "contrastive_loss": contrastive_criterion.state_dict() if contrastive_criterion is not None else None,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "args": vars(args), "nr": nr, "nt": nt, "target_label": args.label}
            torch.save(state, output / f"checkpoint_{epoch}.pt")
            torch.save(state, output / "checkpoint_latest.pt")

    handler.close()
    logger.removeHandler(handler)


if __name__ == "__main__":
    main()
