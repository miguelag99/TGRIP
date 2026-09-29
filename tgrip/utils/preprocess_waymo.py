"""
Preprocess the Waymo Open Dataset v2 (parquet) into a nuScenes-like index.

Only camera data is used. For each segment, frames are subsampled from 10Hz to
2Hz (same 0.5s step as nuScenes keyframes). Camera JPEGs are written as-is and
one index file per split holds nuScenes-shaped sample records:

    {split}_infos.pkl: List[record] sorted by (scene_token, timestamp), where
    record = {
        token, scene_token, timestamp, prev, next,
        ego_pose: 4x4 world_from_vehicle,
        cams: {cam_name: {filename, intrinsic (3x3), rotation (3x3), translation (3,), width, height}},
        anns: [ {token, instance_token, translation (global), size [w,l,h], rotation [w,x,y,z] (global),
                 category_name, visibility_token, dynamic_tag, prev, next} ],
    }

Camera extrinsics are converted to the nuScenes convention (sensor -> ego, optical frame:
x right, y down, z forward). Waymo camera frames are x forward, y left, z up.

Visibility: Waymo has no visibility score. An object is visible (token 4) if its center or any of
its corners projects inside the image of at least one camera, otherwise not visible (token 1).
Occlusion is not taken into account: only objects outside the field of view of all cameras are
removed from supervision. WaymoDB (tgrip/data/dataset/waymo_temporal.py) additionally sets
objects without lidar points (num_lidar_points_in_box == 0) to not visible when loading.

Usage:
    uv run tgrip/utils/preprocess_waymo.py --root ~/Datasets/waymo --split validation
    uv run tgrip/utils/preprocess_waymo.py --root ~/Datasets/waymo --split validation --verify
"""

import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

SUBSAMPLE = 5  # 10Hz -> 2Hz
VEL_THRESHOLD = 0.5  # m/s, same as semantic_data.VEL_THRESHOLD
MIN_DEPTH = 0.1  # Points closer than this to the image plane are not projected.
MATCH_IOU = 0.3  # --verify only: projected 3D box <-> Waymo 2D camera label.

# Waymo CameraName enum. Ordered so that index 1 is the front camera (CAMREF=1).
CAM_NAMES = {
    2: "CAM_FRONT_LEFT",
    1: "CAM_FRONT",
    3: "CAM_FRONT_RIGHT",
    4: "CAM_SIDE_LEFT",
    5: "CAM_SIDE_RIGHT",
}

# Waymo box types (shared by 3D and 2D labels). Vehicles reuse the nuScenes car prompt; other
# classes are kept under Waymo names so that the 'vehicle' category filter drops them.
CATEGORIES = {
    0: "waymo.unknown",
    1: "vehicle.car",
    2: "waymo.pedestrian",
    3: "waymo.sign",
    4: "waymo.cyclist",
}

# nuScenes visibility tokens.
VIS_IN_CAMERA, VIS_NOT_IN_CAMERA = 4, 1

# Columns of the optical frame (x right, y down, z forward) expressed in the Waymo camera frame.
WAYMO_CAM_TO_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

LB = "[LiDARBoxComponent]"
CB = "[CameraBoxComponent]"
CC = "[CameraCalibrationComponent]"
CI = "[CameraImageComponent]"
TS = "key.frame_timestamp_micros"


def read(root, split, component, segment, columns=None, filters=None):
    return pq.read_table(
        os.path.join(root, split, component, f"{segment}.parquet"),
        columns=columns,
        filters=filters,
    )


def yaw_quaternion(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def box_corners(translation, size, yaw):
    """8 corners (3, 8) of a box given nuScenes-style size [w, l, h]."""
    w, l, h = size
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot @ np.stack([x, y, z]) + np.asarray(translation)[:, None]


def project_box(corners_global, world_from_vehicle, cam):
    """Axis-aligned image box [x0, y0, x1, y1] of projected corners, clipped to the image, or None."""
    pts = np.linalg.inv(world_from_vehicle) @ np.vstack([corners_global, np.ones(8)])
    pts = cam["rotation"].T @ (pts[:3] - cam["translation"][:, None])
    if (pts[2] <= MIN_DEPTH).any():
        return None
    uv = cam["intrinsic"] @ (pts / pts[2])
    x0, x1 = np.clip([uv[0].min(), uv[0].max()], 0, cam["width"])
    y0, y1 = np.clip([uv[1].min(), uv[1].max()], 0, cam["height"])
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return [x0, y0, x1, y1]


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def in_camera_fov(corners_global, world_from_vehicle, cam):
    """True if the box center or any of its corners projects inside the image, in front of the camera."""
    pts = np.hstack([corners_global, corners_global.mean(axis=1, keepdims=True)])
    pts = np.linalg.inv(world_from_vehicle) @ np.vstack([pts, np.ones(pts.shape[1])])
    pts = cam["rotation"].T @ (pts[:3] - cam["translation"][:, None])
    in_front = pts[2] > MIN_DEPTH
    uv = cam["intrinsic"] @ (pts[:, in_front] / pts[2, in_front])
    return bool(
        ((uv[0] >= 0) & (uv[0] < cam["width"]) & (uv[1] >= 0) & (uv[1] < cam["height"])).any()
    )


def process_segment(args):
    root, split, out_root, segment = args
    info_path = os.path.join(out_root, split, "infos", f"{segment}.pkl")
    if os.path.exists(info_path):
        return info_path

    # Ego poses and kept timestamps.
    poses = read(root, split, "vehicle_pose", segment).to_pydict()
    ego_poses = {
        ts: np.array(m, dtype=np.float64).reshape(4, 4)
        for ts, m in zip(poses[TS], poses["[VehiclePoseComponent].world_from_vehicle.transform"])
    }
    kept_ts = sorted(ego_poses)[::SUBSAMPLE]

    # Calibration (constant per segment).
    calib = read(root, split, "camera_calibration", segment).to_pydict()
    cams = {}
    for i, cam_id in enumerate(calib["key.camera_name"]):
        K = np.array(
            [
                [calib[f"{CC}.intrinsic.f_u"][i], 0.0, calib[f"{CC}.intrinsic.c_u"][i]],
                [0.0, calib[f"{CC}.intrinsic.f_v"][i], calib[f"{CC}.intrinsic.c_v"][i]],
                [0.0, 0.0, 1.0],
            ]
        )
        ext = np.array(calib[f"{CC}.extrinsic.transform"][i], dtype=np.float64).reshape(4, 4)
        cams[cam_id] = {
            "intrinsic": K,
            "rotation": ext[:3, :3] @ WAYMO_CAM_TO_OPTICAL,
            "translation": ext[:3, 3].copy(),
            "width": calib[f"{CC}.width"][i],
            "height": calib[f"{CC}.height"][i],
        }
    assert set(cams) == set(CAM_NAMES), f"{segment}: cameras {sorted(cams)}"

    # Images: write original JPEG bytes of kept frames. Keep the ego pose at capture time,
    # which is more accurate than the frame pose to project boxes into each camera.
    img_dir = os.path.join(out_root, split, "images", segment)
    os.makedirs(img_dir, exist_ok=True)
    imgs = read(
        root, split, "camera_image", segment,
        columns=[TS, "key.camera_name", f"{CI}.image", f"{CI}.pose.transform"],
        filters=[(TS, "in", kept_ts)],
    ).to_pydict()
    filenames, capture_poses = {}, {}
    for ts, cam_id, data, pose in zip(
        imgs[TS], imgs["key.camera_name"], imgs[f"{CI}.image"], imgs[f"{CI}.pose.transform"]
    ):
        rel = os.path.join(split, "images", segment, f"{ts}_{CAM_NAMES[cam_id]}.jpg")
        with open(os.path.join(out_root, rel), "wb") as f:
            f.write(data)
        filenames[(ts, cam_id)] = rel
        capture_poses[(ts, cam_id)] = np.array(pose, dtype=np.float64).reshape(4, 4)
    assert len(filenames) == len(kept_ts) * len(CAM_NAMES), f"{segment}: missing images"

    # Boxes (vehicle frame) -> global.
    boxes = read(root, split, "lidar_box", segment, filters=[(TS, "in", kept_ts)]).to_pydict()
    anns_per_ts = {ts: {} for ts in kept_ts}
    for i, ts in enumerate(boxes[TS]):
        obj_id = boxes["key.laser_object_id"][i]
        ego = ego_poses[ts]
        center = ego @ np.array([boxes[f"{LB}.box.center.{a}"][i] for a in "xyz"] + [1.0])
        yaw = boxes[f"{LB}.box.heading"][i] + np.arctan2(ego[1, 0], ego[0, 0])
        speed = np.hypot(boxes[f"{LB}.speed.x"][i] or 0.0, boxes[f"{LB}.speed.y"][i] or 0.0)
        corners = box_corners(center[:3], [boxes[f"{LB}.box.size.{a}"][i] for a in "yxz"], yaw)
        visible = any(in_camera_fov(corners, capture_poses[(ts, c)], cams[c]) for c in CAM_NAMES)
        anns_per_ts[ts][obj_id] = {
            "token": f"{segment}_{ts}_{obj_id}",
            "instance_token": f"{segment}_{obj_id}",
            "translation": center[:3].tolist(),
            # nuScenes size is [width, length, height]; Waymo size.x is the length.
            "size": [boxes[f"{LB}.box.size.{a}"][i] for a in "yxz"],
            "rotation": yaw_quaternion(yaw),
            "category_name": CATEGORIES[boxes[f"{LB}.type"][i]],
            "visibility_token": VIS_IN_CAMERA if visible else VIS_NOT_IN_CAMERA,
            "dynamic_tag": "moving" if speed > VEL_THRESHOLD else "stopped",
            "num_lidar_points_in_box": boxes[f"{LB}.num_lidar_points_in_box"][i],
        }

    # Temporal links at 2Hz.
    records = []
    for k, ts in enumerate(kept_ts):
        prev_ts = kept_ts[k - 1] if k > 0 else None
        next_ts = kept_ts[k + 1] if k + 1 < len(kept_ts) else None
        for obj_id, ann in anns_per_ts[ts].items():
            ann["prev"] = anns_per_ts[prev_ts][obj_id]["token"] if prev_ts and obj_id in anns_per_ts[prev_ts] else ""
            ann["next"] = anns_per_ts[next_ts][obj_id]["token"] if next_ts and obj_id in anns_per_ts[next_ts] else ""
        records.append(
            {
                "token": f"{segment}_{ts}",
                "scene_token": segment,
                "timestamp": ts,
                "prev": f"{segment}_{prev_ts}" if prev_ts else "",
                "next": f"{segment}_{next_ts}" if next_ts else "",
                "ego_pose": ego_poses[ts],
                "cams": {
                    name: {**cams[cam_id], "filename": filenames[(ts, cam_id)]}
                    for cam_id, name in CAM_NAMES.items()
                },
                "anns": list(anns_per_ts[ts].values()),
            }
        )

    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path + ".tmp", "wb") as f:
        pickle.dump(records, f)
    os.replace(info_path + ".tmp", info_path)  # Only complete segments are skipped on resume.
    return info_path


def verify(root, out_root, split, bev_ranges, min_front_matches):
    """Checks the stored geometry end-to-end against Waymo's independent 2D camera labels: wrong
    extrinsics, intrinsics, poses or box sizes would leave vehicles right in front of the car
    without an overlapping 2D label. Also reports the field-of-view visibility ratios.

    Checked for several BEV ranges (e.g. the +-15 m short range and the +-50 m long range grids).

    Note: a vehicle is counted as matched if its projected box has IoU >= MATCH_IOU with any 2D
    vehicle label. Occluded vehicles have no label of their own and fail this, which is more
    frequent at long range, hence a lower threshold there. Wrong geometry gives ~0 at any range.
    """
    max_range = max(bev_ranges)
    with open(os.path.join(out_root, f"{split}_infos.pkl"), "rb") as f:
        records = pickle.load(f)
    front_vis, rear_vis, front_match = ({r: [] for r in bev_ranges} for _ in range(3))
    for segment in sorted({r["scene_token"] for r in records}):
        recs = [r for r in records if r["scene_token"] == segment]
        kept_ts = [r["timestamp"] for r in recs]
        cam_box = read(root, split, "camera_box", segment, filters=[(TS, "in", kept_ts)]).to_pydict()
        labels = {}
        for i, (ts, cam_id, t) in enumerate(zip(cam_box[TS], cam_box["key.camera_name"], cam_box[f"{CB}.type"])):
            if t != 1:
                continue
            cx, cy = cam_box[f"{CB}.box.center.x"][i], cam_box[f"{CB}.box.center.y"][i]
            w, h = cam_box[f"{CB}.box.size.x"][i], cam_box[f"{CB}.box.size.y"][i]
            labels.setdefault((ts, cam_id), []).append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        imgs = read(
            root, split, "camera_image", segment,
            columns=[TS, "key.camera_name", f"{CI}.pose.transform"], filters=[(TS, "in", kept_ts)],
        ).to_pydict()
        capture_poses = {
            (ts, c): np.array(m, dtype=np.float64).reshape(4, 4)
            for ts, c, m in zip(imgs[TS], imgs["key.camera_name"], imgs[f"{CI}.pose.transform"])
        }
        for rec in recs:
            vehicle_from_world = np.linalg.inv(rec["ego_pose"])
            for ann in rec["anns"]:
                if ann["category_name"] != "vehicle.car":
                    continue
                x, y, *_ = vehicle_from_world @ np.array(ann["translation"] + [1.0])
                ranges = [r for r in bev_ranges if max(abs(x), abs(y)) < r]
                if not ranges:
                    continue
                for r in ranges:
                    (front_vis if x >= 0 else rear_vis)[r].append(ann["visibility_token"] == VIS_IN_CAMERA)
                if x < 0:
                    continue
                q = ann["rotation"]
                corners = box_corners(ann["translation"], ann["size"], 2 * np.arctan2(q[3], q[0]))
                best = 0.0
                for cam_id, name in CAM_NAMES.items():
                    pbox = project_box(corners, capture_poses[(rec["timestamp"], cam_id)], rec["cams"][name])
                    if pbox is not None:
                        best = max([best] + [iou(pbox, l) for l in labels.get((rec["timestamp"], cam_id), [])])
                for r in ranges:
                    front_match[r].append(best >= MATCH_IOU)
    ratio = lambda v: np.mean(v) if v else 0.0
    failed = []
    for r, min_match in zip(bev_ranges, min_front_matches):
        print(f"+-{r:g} m BEV: vehicles in camera FOV: front half {ratio(front_vis[r]):.3f} "
              f"(n={len(front_vis[r])}), rear half {ratio(rear_vis[r]):.3f} (n={len(rear_vis[r])})")
        print(f"+-{r:g} m BEV: front vehicles matching a 2D label (IoU >= {MATCH_IOU}): "
              f"{ratio(front_match[r]):.3f} (min {min_match})")
        if ratio(front_match[r]) < min_match:
            failed.append(r)
    assert not failed, f"Front vehicles do not match 2D labels at +-{failed} m: check geometry."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Raw Waymo v2 root with {split}/{component}/*.parquet")
    parser.add_argument("--out", default=None, help="Output root. Defaults to <root>/processed")
    parser.add_argument("--split", required=True, choices=["training", "validation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N segments")
    parser.add_argument("--verify", action="store_true", help="Check geometry on the processed index")
    parser.add_argument("--bev_ranges", type=float, nargs="+", default=[15.0, 50.0])
    parser.add_argument("--min_front_matches", type=float, nargs="+", default=[0.95, 0.8])
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    out_root = os.path.expanduser(args.out or os.path.join(root, "processed"))

    if args.verify:
        assert len(args.bev_ranges) == len(args.min_front_matches)
        verify(root, out_root, args.split, args.bev_ranges, args.min_front_matches)
        return

    segments = sorted(
        f[: -len(".parquet")]
        for f in os.listdir(os.path.join(root, args.split, "vehicle_pose"))
        if f.endswith(".parquet")
    )[: args.limit]

    with Pool(args.workers) as pool:
        info_paths = list(
            tqdm(
                pool.imap_unordered(process_segment, [(root, args.split, out_root, s) for s in segments]),
                total=len(segments),
            )
        )

    records = []
    for p in sorted(info_paths):
        with open(p, "rb") as f:
            records.extend(pickle.load(f))
    records.sort(key=lambda r: (r["scene_token"], r["timestamp"]))
    with open(os.path.join(out_root, f"{args.split}_infos.pkl"), "wb") as f:
        pickle.dump(records, f)
    n_anns = sum(len(r["anns"]) for r in records)
    print(f"{args.split}: {len(segments)} segments, {len(records)} samples, {n_anns} annotations")


if __name__ == "__main__":
    main()
