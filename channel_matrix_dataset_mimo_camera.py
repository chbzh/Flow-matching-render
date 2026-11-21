"""Frame-aligned RX/TX/LiDAR to complex MIMO CSI dataset synthesized from CIR paths."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation


class MIMOCSIDataset(Dataset):
    """Reads aligned samples and expresses radio endpoints/LiDAR in one world frame."""
    def __init__(self, sensor_root="/home/largeDisk/bingzheng/DGS/WRF-GSplus-main/MMData/CSI_sensor_data",
                 channel_root="/home/largeDisk/bingzheng/DGS/WRF-GSplus-main/MMData/CSI_data",
                 scenario="Town03_Tjunction_wiz_slope_seed42", channel_scenario="Town03_Tjunction",
                 cav_id=1, num_points=4096, frequency_offset_hz=0.0, seed=1234,
                 rx_pose_key="vehicle_pose", tx_pose_key="rsu_pose", image_size=128):
        self.sensor_dir = Path(sensor_root) / "Town03" / scenario
        self.channel_dir = Path(channel_root) / "Town03" / channel_scenario / f"cav_{cav_id}"
        self.rx_dir, self.tx_dir = self.sensor_dir / f"cav_{cav_id}", self.sensor_dir / "rsu_1"
        self.num_points, self.frequency_offset_hz, self.seed = num_points, frequency_offset_hz, seed
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.cav_id, self.rx_pose_key, self.tx_pose_key = cav_id, rx_pose_key, tx_pose_key
        # Join by numeric frame id; only complete, frame-matched samples enter the dataset.
        rx = {p.stem: p for p in self.rx_dir.glob("*.yaml")}
        tx = {p.stem: p for p in self.tx_dir.glob("*.yaml")}
        pcd = {p.stem: p for p in self.tx_dir.glob("*.pcd")}
        cir = {p.name.removesuffix("_paths.npz"): p for p in self.channel_dir.glob("*_paths.npz")}
        camera_files = [{p.name.split("_camera")[0]: p
                         for p in self.rx_dir.glob(f"*_camera{i}.png")} for i in range(4)]
        keys = set(rx) & set(tx) & set(pcd) & set(cir)
        for camera_index in range(4):
            keys &= set(camera_files[camera_index])
        keys = sorted(keys, key=lambda x: int(x))
        self.samples = [{"frame": k, "rx": rx[k], "tx": tx[k], "pcd": pcd[k], "cir": cir[k],
                         "cameras": [camera_files[i][k] for i in range(4)]} for k in keys]
        if not self.samples:
            raise FileNotFoundError(f"No aligned frames in {self.sensor_dir} and {self.channel_dir}")
        print(f"MIMOCSIDataset: {len(self.samples)} aligned frames with four camera views (CSI synthesized from CIR paths)")

    def _read_camera(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    @staticmethod
    def _pose_position(pose):
        loc = pose["location"]
        return np.array([loc["x"], -loc["y"], loc["z"]], dtype=np.float32)

    @staticmethod
    def _pose_orientation(pose):
        """Flatten the pose rotation after converting CARLA world to Sionna world."""
        rot = pose["rotation"]
        r_carla = Rotation.from_euler(
            "ZYX", [rot["yaw"], rot["pitch"], rot["roll"]], degrees=True
        ).as_matrix()
        reflection = np.diag([1.0, -1.0, 1.0])
        r_world = reflection @ r_carla @ reflection
        return r_world.astype(np.float32).reshape(-1)

    @staticmethod
    def _points_to_world(points_local, pose):
        """Transform CARLA-local LiDAR points into the shared right-handed world frame."""
        rot = pose["rotation"]
        r_carla = Rotation.from_euler(
            "ZYX", [rot["yaw"], rot["pitch"], rot["roll"]], degrees=True
        ).as_matrix().astype(np.float32)
        loc = pose["location"]
        translation_carla = np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float32)
        reflection = np.array([1.0, -1.0, 1.0], dtype=np.float32)
        points_carla_world = points_local @ r_carla.T + translation_carla
        return points_carla_world * reflection

    @staticmethod
    def _read_yaml(path, frame, actor):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if str(data.get("frame", "")).zfill(6) != frame or data.get("actor") != actor:
            raise ValueError(f"Frame/actor mismatch in {path}")
        return data

    @staticmethod
    def _dft_matrix(size):
        row = np.arange(size)[:, None]
        col = np.arange(size)[None, :]
        return np.exp(-2j * np.pi * row * col / size) / np.sqrt(size)

    @staticmethod
    def _complex_to_two_channels(value):
        return torch.from_numpy(np.stack((value.real, value.imag), axis=0).astype(np.float32))

    def _read_channels(self, path):
        with np.load(path) as d:
            if "a" not in d or "tau" not in d:
                raise KeyError(f"{path} must contain a and tau")
            a, tau = np.asarray(d["a"]), np.asarray(d["tau"]).reshape(-1)
        if a.ndim != 7 or any(a.shape[i] != 1 for i in (0, 1, 3, 6)):
            raise ValueError(f"Unsupported path gain shape {a.shape} in {path}")
        gains = a[0, 0, :, 0, :, :, 0]
        if not np.iscomplexobj(gains) or gains.shape[-1] != tau.size:
            raise ValueError(f"Invalid complex path gains/tau in {path}")
        # Synthesize narrowband CSI from CIR paths at the requested baseband
        # frequency offset: H(f) = sum_l a_l exp(-j 2 pi f tau_l).
        h = np.sum(gains * np.exp(-2j * np.pi * self.frequency_offset_hz * tau)[None, None, :], axis=-1)
        # Unitary DFT along RX/TX antenna dimensions, matching
        # MultimodalChannelDataset: channel is angular/sparse-domain CSI;
        # channel_H remains the original spatial-domain CSI.
        f_rx = self._dft_matrix(h.shape[0])
        f_tx = self._dft_matrix(h.shape[1])
        h_sparse = f_rx.conj().T @ h @ f_tx
        return self._complex_to_two_channels(h_sparse), self._complex_to_two_channels(h)

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        rx_y = self._read_yaml(s["rx"], s["frame"], f"cav_{self.cav_id}")
        tx_y = self._read_yaml(s["tx"], s["frame"], "rsu_1")
        # Actor/RSU poses describe the radio endpoints; LiDAR poses describe
        # the sensor origins and are only used to put the RSU cloud in world coordinates.
        rx_pose = rx_y["sensors"][self.rx_pose_key]
        tx_pose = tx_y["sensors"][self.tx_pose_key]
        rx = self._pose_position(rx_pose)
        tx = self._pose_position(tx_pose)
        rx_orientation = self._pose_orientation(rx_pose)
        tx_orientation = self._pose_orientation(tx_pose)
        pcd = o3d.io.read_point_cloud(str(s["pcd"]))
        points = np.asarray(pcd.points, dtype=np.float32)
        if points.size == 0: raise ValueError(f"Empty point cloud: {s['pcd']}")
        lidar_pose = tx_y["sensors"]["lidar_pose"]
        points = self._points_to_world(points, lidar_pose)
        images = torch.stack([self._read_camera(path) for path in s["cameras"]], dim=0)
        rng = np.random.default_rng(self.seed + index)
        if len(points) >= self.num_points:
            points = points[rng.choice(len(points), self.num_points, replace=False)]
        else:
            points = np.concatenate((points, points[rng.choice(len(points), self.num_points-len(points), replace=True)]))
        channel, channel_h = self._read_channels(s["cir"])
        return {"points": torch.from_numpy(np.ascontiguousarray(points)), "images": images,
                "rx_pos": torch.from_numpy(rx),
                "tx_pos": torch.from_numpy(tx),
                "rx_orientation": torch.from_numpy(rx_orientation),
                "tx_orientation": torch.from_numpy(tx_orientation),
                "channel": channel, "channel_H": channel_h,
                "frame_id": s["frame"]}
