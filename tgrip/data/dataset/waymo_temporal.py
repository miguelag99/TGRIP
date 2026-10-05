"""
Temporal Waymo dataloader.

Waymo is preprocessed into nuScenes-shaped records by tgrip/utils/preprocess_waymo.py.
WaymoDB exposes those records through the subset of the nuScenes devkit API used by
NuScenesDataset / TemporalNuScenesDataset, so that all the BEV label generation is shared.
"""

import os
import pickle
import warnings
from typing import List, Tuple

import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from PIL import Image
from pyquaternion import Quaternion

from tgrip.data.dataset.nuscenes_temporal import TemporalNuScenesDataset

WAYMO_SPLITS = {True: "training", False: "validation"}

# Must match CATEGORIES in tgrip/utils/preprocess_waymo.py.
WAYMO_CATEGORIES = [
    "waymo.unknown",
    "vehicle.car",
    "waymo.pedestrian",
    "waymo.sign",
    "waymo.cyclist",
]

# nuScenes visibility token for objects excluded from training and evaluation
# (min_visibility: 2). Must match VIS_NOT_IN_CAMERA in tgrip/utils/preprocess_waymo.py.
NOT_VISIBLE, VISIBLE = 1, 4


def _quaternion(rot: np.ndarray) -> List[float]:
    # Re-orthonormalize: pyquaternion rejects matrices with float noise.
    u, _, vt = np.linalg.svd(rot)
    return list(Quaternion(matrix=u @ vt).elements)


class WaymoDB:
    """Minimal nuScenes-devkit-like API over preprocessed Waymo index files.

    Supported: dataroot, version, category, get(table, token), get_boxes (empty),
    get_sample_data (cameras only).
    Tables: sample_annotation, sample_data, ego_pose, calibrated_sensor, attribute.
    Split samples are loaded on demand with load_split().

    Cameras taller than img_height (Waymo front cameras: 1920x1280) are cropped at the top to
    img_height (side cameras: 1920x886) by TopCropLoader; their principal point is shifted here.

    Visibility: the preprocessed visibility_token only encodes the camera field of view. Objects
    without any lidar point in their box (e.g. fully occluded tracks) are also set to NOT_VISIBLE
    here, so that they are ignored in training and evaluation.

    Observed instances: Waymo has no rear camera, so objects overtaken by the ego leave every camera
    FOV while still being tracked by the model. While `observed_instances` is set (by WaymoDataset,
    per sample), annotations of those instances are returned as VISIBLE in every frame.
    """

    def __init__(self, dataroot: str, img_height: int):
        self.dataroot = dataroot
        self.img_height = img_height
        self.version = "waymo"
        self.category = [{"name": n} for n in WAYMO_CATEGORIES]
        self._samples = {}
        self.observed_instances = set()
        self._tables = {
            t: {}
            for t in ["sample_annotation", "sample_data", "ego_pose", "calibrated_sensor", "attribute"]
        }

    def load_split(self, split: str) -> List[dict]:
        if split in self._samples:
            return self._samples[split]

        with open(os.path.join(self.dataroot, f"{split}_infos.pkl"), "rb") as f:
            records = pickle.load(f)

        samples = []
        for rec in records:
            token = rec["token"]
            self._tables["ego_pose"][token] = {
                "token": token,
                "translation": rec["ego_pose"][:3, 3].tolist(),
                "rotation": _quaternion(rec["ego_pose"][:3, :3]),
            }
            # The sample token doubles as the 'LIDAR_TOP' sample_data token: the ego pose of the frame.
            self._tables["sample_data"][token] = {"token": token, "ego_pose_token": token}

            data = {"LIDAR_TOP": token}
            for cam_name, cam in rec["cams"].items():
                calib_token = f"{rec['scene_token']}_{cam_name}"  # Constant per segment.
                if calib_token not in self._tables["calibrated_sensor"]:
                    top_crop = cam["height"] - self.img_height
                    assert top_crop >= 0, f"{cam_name}: height {cam['height']} < {self.img_height}"
                    intrinsic = cam["intrinsic"].copy()
                    intrinsic[1, 2] -= top_crop
                    self._tables["calibrated_sensor"][calib_token] = {
                        "camera_intrinsic": intrinsic.tolist(),
                        "rotation": _quaternion(cam["rotation"]),
                        "translation": cam["translation"].tolist(),
                        # Original (uncropped) image, used by get_sample_data.
                        "raw_camera_intrinsic": cam["intrinsic"].tolist(),
                        "raw_size": (cam["width"], cam["height"]),
                    }
                sd_token = f"{token}_{cam_name}"
                self._tables["sample_data"][sd_token] = {
                    "token": sd_token,
                    "filename": cam["filename"],
                    "calibrated_sensor_token": calib_token,
                    "ego_pose_token": token,
                }
                data[cam_name] = sd_token

            for ann in rec["anns"]:
                attribute = f"waymo.{ann['dynamic_tag']}"  # Parsed as 'moving' / 'stopped'.
                self._tables["attribute"].setdefault(attribute, {"name": attribute})
                self._tables["sample_annotation"][ann["token"]] = {
                    **ann,
                    "visibility_token": (
                        ann["visibility_token"] if ann["num_lidar_points_in_box"] > 0 else NOT_VISIBLE
                    ),
                    "sample_token": token,
                    "attribute_tokens": [attribute],
                }

            samples.append(
                {
                    "token": token,
                    "scene_token": rec["scene_token"],
                    "timestamp": rec["timestamp"],
                    "prev": rec["prev"],
                    "next": rec["next"],
                    "data": data,
                    "anns": [ann["token"] for ann in rec["anns"]],
                }
            )

        self._samples[split] = samples
        return samples

    def get(self, table: str, token: str) -> dict:
        rec = self._tables[table][token]
        if table == "sample_annotation" and rec["instance_token"] in self.observed_instances:
            return {**rec, "visibility_token": VISIBLE}
        return rec

    def visible_instances(self, samples: List[dict], min_visibility: int) -> set:
        """Instance tokens visible in their own frame in any of the given samples."""
        anns = (self._tables["sample_annotation"][tok] for s in samples for tok in s["anns"])
        return {a["instance_token"] for a in anns if a["visibility_token"] >= min_visibility}

    def get_boxes(self, sample_data_token: str) -> list:
        # Only used for perspective segmentation, which is not supported on Waymo.
        return []

    def get_sample_data(
        self,
        sample_data_token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        selected_anntokens: List[str] = None,
    ) -> Tuple[str, List[Box], np.ndarray]:
        """Same as NuScenes.get_sample_data for camera sample_data: image path, boxes in the camera
        frame that are visible in the image, and intrinsics. Returns the original (uncropped) image
        and its intrinsics. Used to crop objects for the CLIP semantic embeddings."""
        sd = self._tables["sample_data"][sample_data_token]
        cs = self._tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        pose = self._tables["ego_pose"][sd["ego_pose_token"]]
        intrinsic = np.array(cs["raw_camera_intrinsic"])

        boxes = []
        for token in selected_anntokens or []:
            ann = self._tables["sample_annotation"][token]
            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]), token=token)
            # Global -> ego -> camera.
            box.translate(-np.array(pose["translation"]))
            box.rotate(Quaternion(pose["rotation"]).inverse)
            box.translate(-np.array(cs["translation"]))
            box.rotate(Quaternion(cs["rotation"]).inverse)
            if box_in_image(box, intrinsic, cs["raw_size"], vis_level=box_vis_level):
                boxes.append(box)

        return os.path.join(self.dataroot, sd["filename"]), boxes, intrinsic


class TopCropLoader:
    """Crops images at the top (sky) to (W, H). Waymo front cameras are 1920x1280 while side
    cameras are 1920x886; cropping keeps the native aspect ratio and makes all cameras go through
    the same resize / crop augmentation. WaymoDB shifts the principal point accordingly."""

    def __init__(self, loader, W: int, H: int):
        self.loader = loader
        self.mode = loader.mode
        self.size = (W, H)

    def __call__(self, path):
        img = self.loader(path)
        assert isinstance(img, Image.Image), "Waymo cropping requires the PIL image loader."
        if img.size == self.size:
            return img
        W, H = img.size
        assert W == self.size[0] and H >= self.size[1], f"{path}: {img.size}"
        return img.crop((0, H - self.size[1], W, H))


class WaymoDataset(TemporalNuScenesDataset):
    """Temporal Waymo dataset. Expects `nusc` to be a WaymoDB."""

    def __init__(self, *args, **kwargs):
        for key in ["keep_input_persp", "keep_input_hdmap", "keep_input_lidar", "is_lyft"]:
            if kwargs.get(key, False):
                raise NotImplementedError(f"{key}=True is not supported on Waymo.")
        super().__init__(*args, **kwargs)
        assert self.nusc.img_height == self.img_params["H"], "WaymoDB and loader crop must match."
        self.img_loader = TopCropLoader(
            self.img_loader, W=self.img_params["W"], H=self.img_params["H"]
        )

    def __getitem__(self, index):
        # Objects seen in any input camera frame stay valid in every BEV frame of the sample, even
        # after they leave the cameras FOV (no rear camera on Waymo).
        cam_records = [self.ixes[self.indices[index][i]] for i in self.cam_T_index]
        self.nusc.observed_instances = self.nusc.visible_instances(
            cam_records, self.img_params["min_visibility"]
        )
        try:
            return super().__getitem__(index)
        finally:
            self.nusc.observed_instances = set()

    def _get_scenes(self) -> List[str]:
        self.split = WAYMO_SPLITS[self.is_train]
        if self.scene_conditions:
            warnings.warn("scene_conditions filtering is not supported for Waymo; ignoring.")
        return sorted({s["scene_token"] for s in self.nusc.load_split(self.split)})

    def _prepro(self):
        scenes = set(self.scenes)
        samples = [s for s in self.nusc.load_split(self.split) if s["scene_token"] in scenes]
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))
        return samples

    def __str__(self):
        return f"""WaymoDataset: {len(self)} samples. Split: {self.split}."""
