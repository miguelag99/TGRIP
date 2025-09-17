import cv2
from typing import Any, Dict, List, Tuple
from copy import deepcopy
from einops import rearrange

import numpy as np
import numpy.typing as npt
import torch

from .nuscenes_common import DETECTION_CLS, SIGMA
from .nuscenes_temporal import TemporalNuScenesDataset
from .semantic_data import (
    SCENE_TEXT_CONDITIONS, VEL_THRESHOLD, POSITIONAL_CONDITIONS, VELOCITY_CONDITIONS
)
from tgrip.utils.geom import (
    GeomScaler,
    from_corners_to_chw,
    gen_dx_bx,
    get_yawtransfmat_from_mat,
)
from tgrip.utils.imgs import prepare_img_axis
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import affine

THRESHOLD_VALID_CENTERNESS = 0.1

class TextConditionedTemporalNuScenesDataset(TemporalNuScenesDataset):
    def __init__(
        self,
        # Temporal
        cam_T_P: List[List[int]] = [[0, 0]],
        bev_T_P: List[List[int]] = [[0, 0]],
        mode_ref_cam_T: str = "present",
        # Inputs
        keep_input_sampling: bool = False,
        keep_input_detection: bool = False,
        keep_input_centr_offs: bool = False,
        keep_input_hdmap: bool = False,
        keep_input_binimg: bool = True,
        keep_input_offsets_map: bool = False,
        keep_input_lidar: bool = False,
        keep_input_flow_map: bool = False,
        keep_input_instance_bev: bool = False,
        keep_input_persp: bool = False,
        keep_input_semantic_maps: bool = True,
        save_folder: bool = "",
        # Text encoder
        text_encoder = None,
        apply_text_filter: bool = True,
        *args,
        **kwargs,
    ):
        
        super().__init__(
            cam_T_P=cam_T_P,
            bev_T_P=bev_T_P,
            mode_ref_cam_T=mode_ref_cam_T,
            keep_input_sampling=keep_input_sampling,
            keep_input_detection=keep_input_detection,
            keep_input_centr_offs=keep_input_centr_offs,
            keep_input_hdmap=keep_input_hdmap,
            keep_input_binimg=keep_input_binimg,
            keep_input_offsets_map=keep_input_offsets_map,
            keep_input_lidar=keep_input_lidar,
            keep_input_flow_map=keep_input_flow_map,
            keep_input_instance_bev=keep_input_instance_bev,
            keep_input_persp=keep_input_persp,
            save_folder=save_folder,
            *args,
            **kwargs,
        )
        
        self.pose_conditions = POSITIONAL_CONDITIONS
        self.velocity_conditions = VELOCITY_CONDITIONS
        self.text_conditions = SCENE_TEXT_CONDITIONS

        self.text_encoder = text_encoder.to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.apply_text_filter = apply_text_filter
        if apply_text_filter:
            self.keys_to_keep.append("text_condition")
            
        # Embed general text conditions
        for cond in self.text_conditions:
            with torch.no_grad():
                cond['text_embedding'] = self.text_encoder([cond['text_condition']])
        
        self.keep_input_semantic_maps = keep_input_semantic_maps
        if keep_input_semantic_maps:
            self.keys_to_keep.append("semantic_positional_map")
            self.keys_to_keep.append("semantic_positional_map_aug")
            self.keys_to_keep.append("semantic_speed_map")
            self.keys_to_keep.append("semantic_speed_map_aug")

            ## Semantic maps
            # Create text embeddings for conditions
            for cond in self.pose_conditions:
                with torch.no_grad():
                    self.pose_conditions[cond] = self.text_encoder([cond])

            self.full_pos_semantic_map = self._init_positional_semantic_map(
                channels=self.text_conditions[0]['text_embedding'].shape[1],
            )
            
            for status in self.velocity_conditions:
                with torch.no_grad():
                    self.velocity_conditions[status]['embedding'] = self.text_encoder(
                        [self.velocity_conditions[status]['text']]
                    )

        del self.text_encoder # Free memory
                
    def get_bev_related_data(
        self,
        rec,
        egoPout_to_global,
        bev_aug,
        scene_condition: Dict[str, Any] = None,
    ):
        """Return BEV related data.

        Outputs:
            - binimg: (Tensor[torch.uint8]) contains bev segmentation.
            - visibility: (Tensor[torch.uint8]) contains segmentation per visibility level.
            - offsets: (Tensor[torch.float32]) contains distance of objects to the center.
            - centerness: (Tensor[torch.float32]) contains density center map of annotations.
            - bboxes: (Tensor[torch.float32]) contains bounding boxes represented as ordered polygons.
            - binimg_aug: (Tensor[torch.uint8]) contains augmented bev segmentation.
            - classes: (Tensor[torch.uint8]) contains annotated classes.
            - centers: (Tensor[torch.float32]) contains center coordinates.
        """

        # Alias
        h, w = self.nx[0], self.nx[1]

        # Initialize
        # -> Classes
        classes, classes_aug = [], []

        # -> Visibility
        visibility, visibility_aug = np.full((2, h, w), 255, dtype=np.uint8)

        # -> Mobile masks: 0: parked, 1: mobile, 2: stopped, 3: unknown
        mobility, mobility_aug = np.zeros((2, h, w), dtype=np.uint8)
        unrecognized_tag = []

        # -> Offsets
        instance, instance_aug = np.zeros((2, h, w), dtype=np.int32)
        offsets, offsets_aug = torch.full(
            (2, 2, h, w), fill_value=255.0, dtype=torch.float32
        )
        valid_centerness, valid_centerness_aug = np.ones((2, h, w), dtype=np.bool_)

        # -> Offset map
        center_bbox_on_img, center_bbox_on_img_aug = [], []

        x, y = torch.meshgrid(
            torch.arange(h, dtype=torch.float),
            torch.arange(w, dtype=torch.float),
            indexing="xy",
        )

        # -> Centerness
        centerness, centerness_aug = torch.zeros(2, 1, h, w)
        centers, centers_aug = [], []

        # -> Bounding box attributes
        bbox_attr, bbox_attr_aug = [], []

        # -> Bounding boxes
        bboxes, bboxes_aug = {}, {}
        visible_bbox = []
        
        # -> Flow map
        flow_map = torch.zeros(2, h, w)
        
        # -> Semantic maps
        txt_embed_dim = self.text_conditions[0]['text_embedding'].shape[1]
        
        ## Positional semantic map
        semantic_positional_map, semantic_positional_map_aug = (
            torch.zeros(txt_embed_dim, h, w),
            torch.zeros(txt_embed_dim, h, w),
        )
        if 'semantic_positional_map' in self.keys_to_keep:
            semantic_positional_map, semantic_positional_map_aug = (
                self._generate_positional_semantic_map(bev_aug=bev_aug)
            )
        
        ## Movement semantic map
        semantic_speed_map, semantic_speed_map_aug = (
            torch.zeros(txt_embed_dim, h, w),
            torch.zeros(txt_embed_dim, h, w),
        )

        # Are augmentations activated ?
        bool_aug_activated = not np.allclose(bev_aug, np.eye(4))

        egopose_token = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])[
            f"ego_pose_token"
        ]
        inst_egopose = self.nusc.get("ego_pose", egopose_token)
        # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        inst_egopose["size"] = [1.73, 4.084, 1.562]
        inst_egopose["visibility_token"] = 4
        inst_egopose["dynamic_tag"] = 3
        inst_egopose["category_name"] = "vehicle.car"
        inst_egopose["attribute_tokens"] = []
        inst_egopose["instance_token"] = "ego"

        anns = rec["anns"]
        if self.plot_ego:
            anns = anns + [inst_egopose]

        min_vis = self.img_params["min_visibility"]

        # Loop over annotations
        for i, tok in enumerate(anns):
            # Given w.r.t the global coordinate system.
            is_ego = i == len(anns) - 1 and self.plot_ego
            if is_ego:
                inst = tok
            else:
                inst = self.nusc.get("sample_annotation", tok)

            # NuScenesDataset filter:
            if not any([cat in inst["category_name"] for cat in self.filters_cat]):
                continue
            
            # Text condition filtering
            # If there is not a scene condition or the keyword is "all", do not filter.
            if (
                self.apply_text_filter
                and scene_condition is not None
                and scene_condition["keyword"] != "all"
            ):
                # If we filter by dynamic tags, we need to check the attribute tokens.
                if scene_condition.get("keyword") in ["moving", "stopped", "parked"]:
                    if len(inst["attribute_tokens"]) > 0:
                        status = self.nusc.get(
                            "attribute", inst["attribute_tokens"][0]
                        )["name"]
                        if status not in scene_condition["values"]:
                            continue
                    else:
                        # If there are no attribute tokens, we can not filter.
                        continue

                    # Check if the moving objects have a valid velocity.
                    # Avoid conflict due to the lack of velocity attributes in bycicles.
                    if scene_condition["keyword"] == "moving":
                        vel = np.linalg.norm(self.nusc.box_velocity(inst["token"]))
                        if vel < VEL_THRESHOLD:
                            continue

                else:
                    if (inst[scene_condition["filter_by"]]
                        not in scene_condition["values"]
                    ):
                        continue

            # Visibility token, used for detection.
            is_visible = int(inst["visibility_token"]) >= min_vis
            visible_bbox.append(is_visible)

            # Dynamic tag
            if len(inst["attribute_tokens"]) > 0 and (not self.is_lyft):
                assert len(inst["attribute_tokens"]) == 1
                dynamic_tag = self.nusc.get("attribute", inst["attribute_tokens"][0])[
                    "name"
                ]
                dynamic_tag = dynamic_tag.split(".")[-1]
            else:
                dynamic_tag = "other"

            if dynamic_tag in self.map_dynamic_tag.keys():
                inst["dynamic_tag"] = self.map_dynamic_tag[dynamic_tag]
            else:
                if dynamic_tag not in unrecognized_tag:
                    unrecognized_tag.append(dynamic_tag)
                    # print("Unrognized dynamic tag: ", dynamic_tag)
                inst["dynamic_tag"] = self.map_dynamic_tag["other"]

            # Update instance map
            if inst["instance_token"] not in self.inst_map.keys():
                assert (
                    len(self.inst_map) + 1 <= np.iinfo(instance.dtype).max
                ), "Can not encode more instances simultaneously due to precision."
                self.inst_map[inst["instance_token"]] = (
                    len(self.inst_map) + 1
                )  # starts at 1.

            # Bounding boxes:
            (bbox, bbox_aug, bbox_img, bbox_aug_img) = self._get_bbox_region_in_image(
                inst, egoPout_to_global, bev_aug
            )

            bbox, (center, bbox_h, bbox_w), offsets = self._process_bbox_region(
                bbox,bbox_img,visibility,inst,instance,x,y,centerness,SIGMA,offsets,
                mobility,center_bbox_on_img,is_visible, valid_centerness,
            )
                        
            # Semantic maps associated to the instance
            if self.keep_input_semantic_maps and inst["category_name"] in DETECTION_CLS:
                # Velocity conditions
                if len(inst["attribute_tokens"]) > 0:
                    status = self.nusc.get(
                        "attribute", inst["attribute_tokens"][0]
                    )["name"]
                    embed = self.velocity_conditions.get(status, None)
                    embed = embed['embedding'].squeeze(0)                    
                    semantic_speed_map = self._process_semantic_bev_region(
                        bbox_img, semantic_speed_map, embed
                    )
                        
            if bool_aug_activated:
                (
                    bbox_aug,(center_aug, bbox_h_aug, bbox_w_aug),offsets_aug,
                ) = self._process_bbox_region(
                    bbox_aug,bbox_aug_img,visibility_aug,inst,instance_aug,x,y,
                    centerness_aug,SIGMA,offsets_aug,mobility_aug,center_bbox_on_img_aug,
                    is_visible, valid_centerness_aug,
                )

                if (
                    self.keep_input_semantic_maps
                    and inst["category_name"] in DETECTION_CLS
                ):
                    # Velocity conditions
                    if len(inst["attribute_tokens"]) > 0:
                        semantic_speed_map_aug = self._process_semantic_bev_region(
                            bbox_aug_img, semantic_speed_map_aug, embed
                        )
                    
            if is_ego:
                continue

            # Update
            # Objects: only objects that appear inside the image.
            if inst["category_name"] in DETECTION_CLS:
                if self.only_object_center_in:
                    if centers.min() >= -1 and centers.max() <= 1:
                        classes.append(self.class_to_idx[inst["category_name"]])
                        centers.append(center)
                        bbox_attr.append([bbox_h, bbox_w])
                else:
                    classes.append(self.class_to_idx[inst["category_name"]])
                    centers.append(center)
                    bbox_attr.append([bbox_h, bbox_w])

            bboxes[tok] = bbox
            if bool_aug_activated:
                if self.only_object_center_in:
                    if centers_aug.min() >= -1 and centers_aug.max() <= 1:
                        classes_aug.append(self.class_to_idx[inst["category_name"]])
                        centers_aug.append(center_aug)
                        bbox_attr_aug.append([bbox_h_aug, bbox_w_aug])
                else:
                    classes_aug.append(self.class_to_idx[inst["category_name"]])
                    centers_aug.append(center_aug)
                    bbox_attr_aug.append([bbox_h_aug, bbox_w_aug])
                bboxes_aug[tok] = bbox_aug

        # Add egopose bounding box
        (*_, bbox_egopose_img, bbox_egopose_aug_img) = self._get_bbox_region_in_image(
            inst_egopose, egoPout_to_global, bev_aug
        )
        bbox_egopose_img = self.geomscaler.pts_from_spatial_to_img(bbox_egopose_img)
        if bool_aug_activated:
            bbox_egopose_aug_img = self.geomscaler.pts_from_spatial_to_img(
                bbox_egopose_aug_img
            )
        else:
            bbox_egopose_aug_img = bbox_egopose_img

        if not bool_aug_activated:
            # List
            bboxes_aug = deepcopy(bboxes)
            classes_aug = classes.copy()
            center_bbox_on_img_aug = deepcopy(center_bbox_on_img)
            # Numpy
            visibility_aug = visibility.copy()
            mobility_aug = mobility.copy()
            centers_aug = centers.copy()
            bbox_attr_aug = bbox_attr.copy()
            valid_centerness_aug = valid_centerness.copy()
            instance_aug = instance.copy()
            # Torch
            centerness_aug = centerness.clone()
            offsets_aug = offsets.clone()
            semantic_speed_map_aug = semantic_speed_map.clone()
            semantic_positional_map_aug = semantic_positional_map.clone()

        # Can not stack empty list
        if len(centers) > 0:
            classes = torch.tensor(classes, dtype=torch.int64)
            centers = torch.from_numpy(np.stack(centers)).to(torch.float32)
            bbox_attr = torch.from_numpy(np.stack(bbox_attr)).to(torch.float32)

            classes_aug = torch.tensor(classes_aug, dtype=torch.int64)
            centers_aug = torch.from_numpy(np.stack(centers_aug)).to(torch.float32)
            bbox_attr_aug = torch.from_numpy(np.stack(bbox_attr_aug)).to(torch.float32)
        else:
            bbox_attr = torch.empty(0, dtype=torch.float32)
            centers = torch.empty(0, dtype=torch.float32)
            classes = torch.empty(0, dtype=torch.int64)

            bbox_attr_aug = torch.empty(0, dtype=torch.float32)
            centers_aug = torch.empty(0, dtype=torch.float32)
            classes_aug = torch.empty(0, dtype=torch.int64)

        # At least one element.
        if len(bboxes) > 0:
            bboxes = {
                k: torch.from_numpy(np.stack(v)).to(torch.float32)
                for k, v in bboxes.items()
            }
            bboxes_aug = {
                k: torch.from_numpy(np.stack(v)).to(torch.float32)
                for k, v in bboxes_aug.items()
            }

            # Process center_bbox_on_img: filter with visible_bbox
            center_bbox_on_img = torch.stack(center_bbox_on_img).to(torch.float32)
            center_bbox_on_img_aug = torch.stack(center_bbox_on_img_aug).to(
                torch.float32
            )
            offset_map, offset_map_aug = [
                self._get_offset_map_from_center_bbox(
                    torch.stack([x, y], dim=-1), bb[visible_bbox]
                ).permute(2, 0, 1)
                for bb in [center_bbox_on_img, center_bbox_on_img_aug]
            ]
        else:
            bboxes = {"": torch.empty(0, dtype=torch.float32)}
            bboxes_aug = {"": torch.empty(0, dtype=torch.float32)}
            center_bbox_on_img = torch.empty(0, dtype=torch.float32)
            center_bbox_on_img_aug = torch.empty(0, dtype=torch.float32)
            offset_map = torch.full([2, h, w], -1.0, dtype=torch.float32)
            offset_map_aug = torch.full([2, h, w], -1.0, dtype=torch.float32)

        # Ego pose bounding boxes
        bbox_egopose_img = torch.from_numpy(bbox_egopose_img).to(torch.float32)
        bbox_egopose_aug_img = torch.from_numpy(bbox_egopose_aug_img).to(torch.float32)

        # Lidar data
        if self.keep_input_lidar:
            # When using Lyft, some data are not divisible by 5. May be a bug in database.
            lidar_img, lidar_img_aug = self.get_lidar_data(
                rec, egoPout_to_global, bev_aug
            )
        else:
            lidar_img, lidar_img_aug = np.empty((h, w), dtype=np.int32), np.empty(
                (h, w), dtype=np.int32
            )
        # Prepare outputs
        (
            visibility,
            visibility_aug,
            mobility,
            mobility_aug,
            valid_centerness,
            valid_centerness_aug,
            lidar_img,
            lidar_img_aug,
            instance,
            instance_aug,
        ) = [
            torch.from_numpy(x).unsqueeze(0)
            for x in [
                visibility,
                visibility_aug,
                mobility,
                mobility_aug,
                valid_centerness,
                valid_centerness_aug,
                lidar_img,
                lidar_img_aug,
                instance,
                instance_aug,
            ]
        ]

        # Infer binimg from visibility
        binimg, binimg_aug = [
            torch.floor(1 - x // 255) for x in [visibility, visibility_aug]
        ]

        # BEV validity.
        valid_binimg = visibility >= min_vis
        valid_binimg_aug = visibility_aug >= min_vis
        valid_centerness = valid_centerness.bool()
        valid_centerness_aug = valid_centerness_aug.bool()

        # Change axes: space: (X: bottom, Y: right) -> image: (X: right, Y: bottom)
        [
            visibility,
            visibility_aug,
            mobility,
            mobility_aug,
            offsets,
            offsets_aug,
            centerness,
            centerness_aug,
            binimg,
            binimg_aug,
            valid_binimg,
            valid_binimg_aug,
            offset_map,
            offset_map_aug,
            valid_centerness,
            valid_centerness_aug,
            lidar_img,
            lidar_img_aug,
            instance,
            instance_aug,
            semantic_positional_map,
            semantic_positional_map_aug,
            semantic_speed_map,
            semantic_speed_map_aug,
        ] = [
            prepare_img_axis(x, self.to_cam_ref)
            for x in [
                visibility,
                visibility_aug,
                mobility,
                mobility_aug,
                offsets,
                offsets_aug,
                centerness,
                centerness_aug,
                binimg,
                binimg_aug,
                valid_binimg,
                valid_binimg_aug,
                offset_map,
                offset_map_aug,
                valid_centerness,
                valid_centerness_aug,
                lidar_img,
                lidar_img_aug,
                instance,
                instance_aug,
                semantic_positional_map,
                semantic_positional_map_aug,            
                semantic_speed_map,
                semantic_speed_map_aug,
            ]
        ]

        return {
            "binimg": binimg,
            "binimg_aug": binimg_aug,
            "valid_binimg": valid_binimg,
            "valid_binimg_aug": valid_binimg_aug,
            "visibility": visibility,
            "visibility_aug": visibility_aug,
            "mobility": mobility,
            "mobility_aug": mobility_aug,
            "offsets": offsets,
            "offsets_aug": offsets_aug,
            "lidar_img": lidar_img,
            "lidar_img_aug": lidar_img_aug,
            "valid_centerness": valid_centerness,
            "valid_centerness_aug": valid_centerness_aug,
            "offsets_map": offset_map,
            "offsets_map_aug": offset_map_aug,
            "centerness": centerness,
            "centerness_aug": centerness_aug,
            "bboxes": bboxes,
            "bboxes_aug": bboxes_aug,
            "bbox_egopose": bbox_egopose_img,
            "bbox_egopose_aug": bbox_egopose_aug_img,
            "centers": centers,
            "centers_aug": centers_aug,
            "classes": classes,
            "classes_aug": classes_aug,
            "bbox_attr": bbox_attr,
            "bbox_attr_aug": bbox_attr_aug,
            "instance": instance,
            "instance_aug": instance_aug,
            "semantic_positional_map": semantic_positional_map,
            "semantic_positional_map_aug": semantic_positional_map_aug,
            "semantic_speed_map": semantic_speed_map,
            "semantic_speed_map_aug": semantic_speed_map_aug,
        }
    
    def _init_positional_semantic_map(
        self,
        channels: int = 512,
    ) -> torch.Tensor:
        """Generate a positional semantic map based on the text conditions.

        6 regions are defined following nuScenesQA specs:
            front, front_left, front_right, back, back_left, back_right.

        To avoid invalid values with data augmentation, we create the map with double
        extension and then crop it during data loading.
        
        Args:
            channels (int): Number of channels for the positional semantic map.
        Returns:
            torch.Tensor: Positional semantic map.
        """
        # Alias
        h, w = 2*self.nx[0], 2*self.nx[1]

        # Initialize
        positional_semantic = torch.zeros(channels, h, w)

        # Create a BEV semantic map filled with positional semantic embeddings
        H, W = h, w
        cx, cy = W // 2, H // 2
        
        x = np.arange(H)
        y = np.arange(W)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        X = X * self.grid['xbound'][2] + (self.grid['xbound'][2] / 2 + (2 * self.grid['xbound'][0]))
        Y = Y * self.grid['ybound'][2] + (self.grid['ybound'][2] / 2 + (2 * self.grid['ybound'][0]))
        XY = np.stack([X, Y, np.zeros_like(X), np.ones_like(X)]) # [H, W, 4]
        XY = XY.reshape(4, -1)  # [4, H*W]
        

        # Fill positional_semantic according to angle θ using real-world coordinates from XY
        for idx in range(H * W):
            # XY[:, idx] contains [X, Y, 0, 1] in meters
            x_m = XY[0, idx]
            y_m = XY[1, idx]

            theta = np.degrees(np.arctan2(x_m, y_m))  # θ=0 is right (X axis), CCW positive

            if -30 < theta <= 30:
                key = 'front'
            elif 30 < theta <= 90:
                key = 'front_left'
            elif 90 < theta <= 150:
                key = 'back_left'
            elif 150 < theta or theta <= -150:
                key = 'back'
            elif -90 < theta <= -30:
                key = 'front_right'
            elif -150 < theta <= -90:
                key = 'back_right'
            else:
                key = 'back'

            y = idx // W
            x = idx % W
            positional_semantic[:, y, x] = self.pose_conditions[key][0]
                
        return positional_semantic
            
        
    
    def _generate_positional_semantic_map(
        self,
        bev_aug: np.ndarray = np.eye(4),
    ) -> torch.Tensor:
        """Crop and transform the positional semantic map based on the BEV augmentation.
        Args:
            channels (int): Number of channels for the positional semantic map.
            bev_aug (np.ndarray): Augmentation matrix moving the bev. Impacts the BEV.

        Returns:
            torch.Tensor: Positional semantic map.
        """
        # Alias
        h, w = 2*self.nx[0], 2*self.nx[1]

        # Initialize
        positional_semantic_aug = torch.zeros_like(self.full_pos_semantic_map)

        # Are augmentations activated ?
        bool_aug_activated = not np.allclose(bev_aug, np.eye(4))

        # flattened_positional_semantic = positional_semantic.view(channels, -1).T  # [H*W, C]
        positional_semantic_aug = self.full_pos_semantic_map.clone()
        
        if bool_aug_activated:
            # Method 1: aug in 3d space            
            # aug_XY = torch.Tensor(bev_aug @ XY).T[:, :2]  # [H*W, 2]
            
            # dx, bx, nx = gen_dx_bx(self.grid['xbound'], self.grid['ybound'], self.grid['zbound'])
            # bev_aug_XY = (aug_XY - bx[:2] + (dx[:2] / 2.0)) / dx[:2]
            # # bev_aug_XY[:, 1] = -(bev_aug_XY[:, 1] - H)

            # for idx in range(bev_aug_XY.shape[0]):
            #     # import pdb; pdb.set_trace()
            #     x_idx = int(bev_aug_XY[idx, 0])
            #     y_idx = int(bev_aug_XY[idx, 1])
            #     if 0 <= x_idx < W and 0 <= y_idx < H:
            #         positional_semantic_aug[:, x_idx, y_idx] = flattened_positional_semantic[idx]
            
            # Method 2: use img transformation
            # Get rotation angle from bev_aug matrix
            rot = R.from_matrix(bev_aug[:3, :3])
            yaw = -rot.as_euler('xyz')[2]  # Rotation around Z axis (yaw)
            translation = [-int(bev_aug[0, 3]), -int(bev_aug[1, 3])]
            
            
            positional_semantic_aug = affine(
                positional_semantic_aug,
                angle=np.degrees(yaw),
                translate=translation,
                scale=1.0,
                shear=0
            )

        size_filter = (self.nx[0] // 2, self.nx[1] // 2)
        positional_semantic = self.full_pos_semantic_map[
            :, size_filter[0] : -size_filter[0], size_filter[1] : -size_filter[1]
        ]
        positional_semantic_aug = positional_semantic_aug[
            :, size_filter[0] : -size_filter[0], size_filter[1] : -size_filter[1]
        ]

        return positional_semantic, positional_semantic_aug

    def _process_semantic_bev_region(
        self,
        bbox_img,
        semantic_map,
        text_embedding,
    ) -> torch.Tensor:
        
        # -> round
        poly_region_img_rd = self.geomscaler.pts_from_spatial_to_img(bbox_img)
        poly_region_img_rd = (np.round(poly_region_img_rd)).astype(np.int32)
        
        mask = np.zeros(semantic_map.shape[1:], dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly_region_img_rd, 1)
        semantic_map[:, mask == 1] = text_embedding.unsqueeze(-1).to(semantic_map.device)

        return semantic_map

    def _get_outputs_bev(
        self,
        bev_records_T: List[Tuple[int, int]],
        egoPout_to_global: npt.NDArray,
        bev_aug: npt.NDArray,
        condition: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Get the BEV related outputs.

        Args:
            bev_records_T (List[Tuple[int, int]]): List of output time and pose.
            egoPout_to_global (npt.NDArray): Matrix from world to recorded ego reference frame.
            egoTout_to_seq (npt.NDArray): Matrix from one ego (time) to another ego (pose) reference frame.
            seq_aug (npt.NDArray): Augmentation matrix moving the sequence. Does not impact the BEV.
            bev_aug (npt.NDArray): Augmentation matrix moving the bev. Impacts the BEV.

        Returns:
            List[Dict[str, Any]]: List of BEV related outputs.
        """
        data_bev = []
        tokens = []

        for i, rec in enumerate(bev_records_T):
            tokens.append(rec["token"])
            out_bev_dict = self.get_bev_related_data(
                rec=rec,
                egoPout_to_global=egoPout_to_global[i],
                bev_aug=bev_aug[i],  # from query to query aug.
                scene_condition=condition,
            )

            out_bev_dict.update({"tokens": tokens})

            if self.with_hdmap:
                out_bev_dict.update(self.get_map_related_data(rec, bev_aug[i]))

            data_bev.append(out_bev_dict)

        final_instance_map = self.inst_map
        
        # Reset instance mapping.
        self.inst_map = {}
        self.center_map = {}

        return data_bev, final_instance_map
            
    def __getitem__(self, index):
        # Records of the sequence
        records = np.array([self.ixes[i] for i in self.indices[index]])

        # Input records:
        cam_records = records[self.cam_T_index]
        present_record = records[self.present_index][0]

        # Get camera inputs
        out_dict, *_ = self._get_inputs_cam(cam_records)
        
        # Input egomotion
        out_dict['future_egomotion'] = self.get_future_egomotion(self.indices[index])

        # Get sequence-aug transformation matrix
        self._save_egoTin_to_seq(out_dict, cam_records, present_record)

        # Bev GT:
        # Get BEV records:
        bev_records_T = records[self.bev_T_index]
        bev_records_P = records[self.bev_P_index]

        # Get query-aug transformation matrix
        bev_aug = self._get_inputs_bevaug(bev_records_P)
        self._save_bev_aug(out_dict, bev_aug)

        # -> Ego motion: output reference timestep.
        egoPout_to_global = np.stack(
            [
                get_yawtransfmat_from_mat(self._get_ego_to_global(rec_p))
                for rec_p in bev_records_P
            ]
        )

        self._save_egoreftoego_out_dict(out_dict, bev_records_T, bev_records_P)

        # Get bev outputs
        # Randomly select a text condition
        if self.apply_text_filter:
            random_condition = np.random.choice(self.text_conditions)
            out_dict['text_condition'] = random_condition['text_condition']
        else:
            random_condition = None

        # Get bev outputs with the selected condition
        data_bev, instance_map = self._get_outputs_bev(
            bev_records_T, egoPout_to_global, bev_aug, condition=random_condition
        )
        
        predict_keys = list(data_bev[0].keys())
    
        # Get flow label if required
        if 'flow_map' in self.keys_to_keep:
            flow_label, flow_label_aug = self._get_flow_label(
                data_bev, instance_map, instance_map
            )
            out_dict.update({"flow_map": flow_label})
            out_dict.update({"flow_map_aug": flow_label_aug})

        # Prepare
        out_dict = self._prepare_out_dict(out_dict, data_bev, predict_keys)
        return out_dict

   