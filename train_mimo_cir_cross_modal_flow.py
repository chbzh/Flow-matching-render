"""Liang et al. cross-modal flow for CSI, trained on the con2 Town03 split.

Implements Figures 3–4, equations (14)–(19), and Algorithm 2 of
Environment-Aware Channel Inference via Cross-Modal Flow (TMC 2026).
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
from tqdm import tqdm

from train_mimo_cir_triplane_nerf2_128_flowmatching_renderdiffusion_csi_complex_scatter_re_con2 import (
    make_loaders,
    metric,
    target_from,
)


def sinusoidal_encoding(value, width):
    """Transformer sinusoidal encoding applied to each scalar input."""
    frequencies = torch.arange(0, width, 2, device=value.device, dtype=value.dtype)
    phase = value.unsqueeze(-1) * torch.exp(frequencies * (-math.log(10000.0) / width))
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class MultimodalStochasticEncoder(nn.Module):
    def __init__(self, nr, nt, image_size):
        super().__init__()
        self.nr, self.nt = nr, nt
        feature_dim = 2 * nr * nt
        # The paper specifies the layer counts and shapes at the fusion heads.
        # The intermediate widths, kernel sizes, and head count are unspecified.
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (image_size // 8) ** 2, feature_dim),
        )
        self.point_features = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
        )
        self.point_projection = nn.Linear(256, feature_dim)
        self.gps = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, feature_dim))
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.mean = nn.Conv2d(2, 2, 3, padding=1)
        self.log_variance = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, image, points, gps):
        image_features = self.image(image)
        lidar_features = self.point_projection(self.point_features(points.transpose(1, 2)).amax(dim=-1))
        gps_features = self.gps(sinusoidal_encoding(gps, 256).flatten(1))
        tokens = torch.stack((image_features, lidar_features, gps_features), dim=1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        fused = attended.mean(dim=1).relu().reshape(-1, 2, self.nr, self.nt)
        return self.mean(fused), self.log_variance(fused)


class TimeConditionedConvolutions(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_projection = nn.Linear(time_dim, in_channels)
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
        )

    def forward(self, feature, time_embedding):
        return self.convolutions(feature + self.time_projection(time_embedding)[:, :, None, None])


class NeuralVelocityField(nn.Module):
    """Figure 4 U-Net with additive time embeddings and post-pooling skips."""

    def __init__(self):
        super().__init__()
        self.time_dim = 256
        self.encoders = nn.ModuleList(
            TimeConditionedConvolutions(input_width, width, self.time_dim)
            for input_width, width in ((2, 128), (128, 256), (256, 512))
        )
        self.bottleneck = TimeConditionedConvolutions(512, 1024, self.time_dim)
        self.decoders = nn.ModuleList((
            TimeConditionedConvolutions(1024 + 512, 512, self.time_dim),
            TimeConditionedConvolutions(512 + 256, 256, self.time_dim),
            TimeConditionedConvolutions(256 + 128, 128, self.time_dim),
        ))
        self.upsamplers = nn.ModuleList((
            nn.ConvTranspose2d(512, 512, 2, stride=2),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ConvTranspose2d(128, 2, 2, stride=2),
        ))

    def forward(self, state, t):
        time_embedding = sinusoidal_encoding(t, self.time_dim)
        skips = []
        for encoder in self.encoders:
            state = F.max_pool2d(encoder(state, time_embedding), 2)
            skips.append(state)
        state = self.bottleneck(state, time_embedding)
        for decoder, upsample, skip in zip(self.decoders, self.upsamplers, reversed(skips)):
            state = upsample(decoder(torch.cat((skip, state), dim=1), time_embedding))
        return state


class CrossModalFlow(nn.Module):
    def __init__(self, nr, nt, image_size):
        super().__init__()
        if nr % 8 or nt % 8 or image_size < 8:
            raise ValueError("CSI dimensions must be divisible by 8 and image size must be at least 8")
        self.encoder = MultimodalStochasticEncoder(nr, nt, image_size)
        self.velocity = NeuralVelocityField()

    def training_state(self, sensors, target):
        mean, log_variance = self.encoder(*sensors)
        x0 = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        x_t = torch.lerp(x0, target, t[:, None, None, None])
        velocity = self.velocity(x_t, t)
        return x0, t, x_t, velocity, mean, log_variance

    def integrate(self, mean, steps):
        """Euler startup and second-order Adams–Bashforth, Eqs. (29)–(30)."""
        step = 1.0 / steps
        previous_velocity = self.velocity(mean, mean.new_zeros(mean.shape[0]))
        state = mean + step * previous_velocity
        for k in range(1, steps):
            velocity = self.velocity(state, state.new_full((state.shape[0],), k * step))
            state = state + step * (1.5 * velocity - 0.5 * previous_velocity)
            previous_velocity = velocity
        return state

    def sample(self, sensors, steps=7):
        mean, _ = self.encoder(*sensors)
        return self.integrate(mean, steps)


class ModalityAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, x0, target, mean, log_variance):
        source = F.normalize(x0.flatten(1), dim=1)
        truth = F.normalize(target.flatten(1), dim=1)
        logits = (source @ truth.T) / self.temperature
        contrastive = F.cross_entropy(logits, torch.arange(x0.shape[0], device=x0.device))
        kl = (mean.square() + log_variance.exp() - 1.0 - log_variance).flatten(1).sum(1).mean()
        return contrastive + 1e-4 * kl, contrastive


def sensing_inputs(batch, device):
    image, points, rx, tx = (batch[key].to(device, non_blocking=True)
                             for key in ("rsu_image", "points", "rx_pos", "tx_pos"))
    return image, points - tx[:, None], (rx - tx)[:, :2]


def reverse_angle_bins(csi):
    # The dataset and paper use opposite Fourier signs. Negating the DFT
    # indices on both axes (keeping DC in place) maps either convention to the other.
    return csi.flip((-2, -1)).roll((1, 1), dims=(-2, -1))


def paper_angular(csi, label):
    if label == "channel":
        return reverse_angle_bins(csi)
    spatial = torch.complex(csi[:, 0], csi[:, 1])
    angular = torch.fft.ifft(torch.fft.fft(spatial, dim=-2, norm="ortho"), dim=-1, norm="ortho")
    return torch.stack((angular.real, angular.imag), dim=1)


def output_csi(angular, label):
    if label == "channel":
        return reverse_angle_bins(angular)
    angular = torch.complex(angular[:, 0], angular[:, 1])
    spatial = torch.fft.fft(torch.fft.ifft(angular, dim=-2, norm="ortho"), dim=-1, norm="ortho")
    return torch.stack((spatial.real, spatial.imag), dim=1)


def evaluate(model, train_loader, val_loader, device, args, log_dir, logger, epoch):
    model.eval()
    with torch.inference_mode():
        if args.overfit_samples:
            values = {mode: ([], []) for mode in ("initializer", "t0", "sample")}
            for batch in train_loader:
                target = target_from(batch, args.label, normalize=args.normalize_csi).to(device)
                mean, _ = model.encoder(*sensing_inputs(batch, device))
                predictions = (mean, model.integrate(mean, 1), model.integrate(mean, args.inference_steps))
                for (cosines, nmses), prediction in zip(values.values(), predictions):
                    cos, nmse = metric(output_csi(prediction, args.label), target)
                    cosines.extend(cos.cpu().tolist())
                    nmses.extend(nmse.cpu().tolist())
            result = {mode: dict(cosine=float(np.mean(cosines)), nmse_db=float(np.mean(nmses)))
                      for mode, (cosines, nmses) in values.items()}
            print("[Overfit diagnostic] " + json.dumps(result))
            logger.info("Overfit diagnostic %s", json.dumps(result))
            return result["sample"]["cosine"], result["sample"]["nmse_db"]

        cos_all, nmse_all = [], []
        save_dir = log_dir / "test" / str(epoch)
        image_dir = save_dir / "pred_csi"
        image_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "all_cos_nmse.txt").open("w", encoding="utf-8") as metrics_file:
            metrics_file.write("index\tframe_id\tcosine_similarity\tnmse_db\n")
            sample_no = 0
            for batch in tqdm(val_loader, desc=f"Test {epoch}", leave=False):
                target = target_from(batch, args.label, normalize=args.normalize_csi).to(device)
                angular = model.sample(sensing_inputs(batch, device), args.inference_steps)
                pred = output_csi(angular, args.label)
                cos, nmse = metric(pred, target)
                cos_values, nmse_values = cos.cpu().tolist(), nmse.cpu().tolist()
                cos_all.extend(cos_values)
                nmse_all.extend(nmse_values)
                for frame, pn, tn, c, n in zip(
                    batch["frame_id"], pred.cpu().numpy(), target.cpu().numpy(), cos_values, nmse_values
                ):
                    stem = f"{sample_no:05d}_{frame}"
                    metrics_file.write(f"{sample_no}\t{frame}\t{c:.8f}\t{n:.8f}\n")
                    # An empty triplane array preserves con2's export schema.
                    np.savez_compressed(save_dir / f"{stem}.npz", prediction=pn,
                                        ground_truth=tn, frame_id=frame, label=args.label,
                                        triplane=np.empty(0, dtype=np.float32))
                    pm, tm = np.hypot(pn[0], pn[1]), np.hypot(tn[0], tn[1])
                    low = min(float(pm.min()), float(tm.min()))
                    high = max(float(pm.max()), float(tm.max()), low + 1e-12)
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    img = ax[0].imshow(tm, origin="lower", aspect="auto", cmap="viridis",
                                       vmin=low, vmax=high)
                    ax[0].set_title("Ground truth CSI |H|")
                    ax[1].imshow(pm, origin="lower", aspect="auto", cmap="viridis",
                                 vmin=low, vmax=high)
                    ax[1].set_title("Iterative flow-predicted CSI |H|")
                    fig.subplots_adjust(bottom=0.2, wspace=0.12)
                    colorbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.035])
                    fig.colorbar(img, cax=colorbar_ax, orientation="horizontal")
                    fig.savefig(image_dir / f"{stem}.png", dpi=150)
                    plt.close(fig)
                    sample_no += 1
        return float(np.mean(cos_all)), float(np.mean(nmse_all))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-root", default="/root/autodl-tmp/Document/cbz/CSI_sensor_data")
    parser.add_argument("--channel-root", default="/root/autodl-tmp/Document/cbz/CSI_data")
    parser.add_argument("--scenario", default="Town03_Tjunction_wiz_slope_seed42")
    parser.add_argument("--channel-scenario", default="Town03_Tjunction")
    parser.add_argument("--rx-pose-key", choices=("vehicle_pose", "lidar_pose"), default="lidar_pose")
    parser.add_argument("--tx-pose-key", choices=("rsu_pose", "lidar_pose"), default="lidar_pose")
    parser.add_argument("--label", choices=("channel", "channel_H"), default="channel")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--cav-num-points", type=int, default=24000)
    parser.add_argument("--use-all-cavs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-csi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--lr-start-epoch", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--inference-steps", type=int, default=7)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency-offset-hz", type=float, default=0.0)
    parser.add_argument("--output", default="/root/autodl-tmp/Document/cbz/WRF-GSplus-main-4angents/runs/mimo_cir_cross_modal_flow")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.inference_steps, args.test_every, args.save_every,
           args.lr_step_size, args.num_points, args.cav_num_points) < 1:
        parser.error("Epochs, batch size, inference steps, intervals and point counts must be positive")
    if args.overfit_samples < 0 or args.num_workers < 0 or args.image_size < 8:
        parser.error("Invalid overfit sample count, worker count or image size")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, train_loader, val_loader, train_ids, val_ids, val_frames = make_loaders(args, device)
    first_train_path = next(info["cir"] for ds in datasets for info in ds.samples
                            if info["frame"] not in val_frames)
    first = datasets[0]._read_channels(first_train_path)[0]
    nr, nt = first.shape[-2:]
    model = CrossModalFlow(nr, nt, args.image_size).to(device)
    alignment = ModalityAlignment().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters()},
        {"params": alignment.parameters(), "weight_decay": 0.0},
    ], lr=args.lr, weight_decay=1e-4)
    train_parameters = [*model.parameters(), *alignment.parameters()]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) / timestamp
    log_dir = Path("/root/autodl-tmp/Document/cbz/WRF-GSplus-main-4angents/logs") / f"mimo_cir_cross_modal_flow_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cross_modal_flow_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_dir / "logger.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("CSI=%dx%d, train=%d, val=%d, CAVs=%s, device=%s",
                nr, nt, len(train_ids), len(val_ids), "1-4" if args.use_all_cavs else "2", device)
    geometry_summary = (f"label={args.label}, RX pose={args.rx_pose_key}, TX pose={args.tx_pose_key}, "
                        "RSU RGB (single view), RSU LiDAR, relative RX XY, "
                        f"velocity field=angular CSI and t, AB2 steps={args.inference_steps}, "
                        f"target={'peak-normalized-CSI' if args.normalize_csi else 'raw-CSI'}")
    print(geometry_summary)
    logger.info(geometry_summary)
    logger.info("Run directory %s", output)

    def checkpoint_state(epoch):
        return {"epoch": epoch, "model": model.state_dict(), "contrastive_loss": alignment.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "args": vars(args), "nr": nr, "nt": nt, "target_label": args.label}

    history, best_nmse = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        csi_losses, cos_values, nmse_values, sampled_t_values = [], [], [], []
        total_losses, energy_ratios = [], []
        contrastive_losses = []
        progress = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            target = target_from(batch, args.label, normalize=args.normalize_csi).to(device)
            angular = paper_angular(target, args.label)
            x0, sampled_t, x_t, velocity, mean, log_variance = model.training_state(
                sensing_inputs(batch, device), angular)
            cfm_loss = (velocity - (angular - x0)).flatten(1).square().sum(1).mean()
            alignment_loss, contrastive_loss = alignment(x0, angular, mean, log_variance)
            loss = cfm_loss + alignment_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 1.0)
            optimizer.step()

            with torch.no_grad():
                endpoint = x_t + (1.0 - sampled_t[:, None, None, None]) * velocity
                prediction = output_csi(endpoint, args.label)
                cos, nmse = metric(prediction, target)
                per_sample_mse = (prediction - target).flatten(1).square().mean(1)
                weighted_mse = (per_sample_mse / (1.0 + sampled_t).square()).mean()
                csi_losses.append(per_sample_mse.mean().item())
                total_losses.append(loss.item())
                contrastive_losses.append(contrastive_loss.item())
                energy_ratios.extend((x0.flatten(1).square().sum(1) /
                                      angular.flatten(1).square().sum(1).clamp_min(1e-12)).cpu().tolist())
                sampled_t_values.extend(sampled_t.cpu().tolist())
                cos_values.extend(cos.cpu().tolist())
                nmse_values.extend(nmse.cpu().tolist())
                progress.set_postfix(weighted_mse=f"{weighted_mse.item():.4g}",
                                     t=f"{sampled_t.mean().item():.3f}",
                                     cosine=f"{cos.mean().item():.4f}",
                                     nmse_db=f"{nmse.mean().item():.2f}")

        record = {"epoch": epoch, "train_loss": float(np.mean(total_losses)),
                  "train_csi_mse": float(np.mean(csi_losses)),
                  "train_contrastive_loss": float(np.mean(contrastive_losses)),
                  "contrastive_temperature": alignment.temperature.item(),
                  "train_mean_sampled_t": float(np.mean(sampled_t_values)),
                  "train_cosine": float(np.mean(cos_values)),
                  "train_nmse_db": float(np.mean(nmse_values)),
                  "train_x0_target_energy_ratio": float(np.mean(energy_ratios)),
                  "lr": optimizer.param_groups[0]["lr"]}
        print(json.dumps(record))
        if epoch % args.test_every == 0 or epoch == args.epochs:
            test_cos, test_nmse = evaluate(model, train_loader, val_loader, device, args, log_dir, logger, epoch)
            record.update(test_cosine=test_cos, test_nmse_db=test_nmse)
            print(f"Epoch {epoch}: test cosine={test_cos:.5f}, NMSE={test_nmse:.3f} dB")
            logger.info("Epoch %d test cosine=%.6f NMSE=%.3f dB", epoch, test_cos, test_nmse)
            if test_nmse < best_nmse:
                best_nmse = test_nmse
                torch.save(checkpoint_state(epoch), output / "checkpoint_best.pt")
        if epoch >= args.lr_start_epoch and (epoch - args.lr_start_epoch) % args.lr_step_size == 0:
            scheduler.step()
            record["lr"] = optimizer.param_groups[0]["lr"]
        history.append(record)
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = checkpoint_state(epoch)
            torch.save(state, output / f"checkpoint_{epoch}.pt")
            torch.save(state, output / "checkpoint_latest.pt")

    handler.close()
    logger.removeHandler(handler)


if __name__ == "__main__":
    main()
train_mimo_cir_cross_modal_flow.py
