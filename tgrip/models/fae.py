import time
from collections import OrderedDict
from functools import partial

import numpy as np
import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from pyquaternion import Quaternion
from types import SimpleNamespace

from transformers import SegformerConfig, SegformerForSemanticSegmentation
from efficientnet_pytorch import EfficientNet

from tgrip.models.heads import SemanticHead

## Wrapper ##

def parse_fast_and_efficient():
    cfg = fae_cfg
    hparams = namespace_to_dict(fae_cfg)
    
    # Load checkpoint, but only load matching keys (e.g., for new head)
    checkpoint = torch.load(
        os.path.join(cfg.PRETRAINED.PATH, cfg.PRETRAINED.CKPT), map_location="cpu",
        weights_only=False
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v

    module = TrainingModule(hparams, cfg)
    model_state = module.model.state_dict()
    # Only load matching keys
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered_state_dict)
    module.model.load_state_dict(model_state, strict=False)
        
    return cfg, hparams, module.model

## Configs ##

fae_cfg = SimpleNamespace(
    
    LIDAR_SUPERVISION = False,
    
    # Pretrainied or resume training configuration
    # If LOAD_WEIGHTS is True, then the model weights will be loaded from the path 
    # specified in CKPT 
    # If both LOAD_WEIGHTS and RESUME_TRAINING are True, then the model weights will
    # be loaded and 
    # training will resume mantaining the optimizer and shceduler states.
    
    PRETRAINED = SimpleNamespace(
        LOAD_WEIGHTS = False,
        RESUME_TRAINING = False,
        PATH = '/home/perception/workspace/',
        CKPT = 'checkpoints/B0-short.ckpt',
    ),
    
    DATASET = SimpleNamespace(
        DATAROOT = '/home/perception/Datasets/nuscenes/',
        VERSION = 'v1.0-trainval',
        NAME = 'nuscenes',
        IGNORE_INDEX = 255,  # Ignore index when creating flow/offset labels
        FILTER_INVISIBLE_VEHICLES = True,  # Filter vehicles not visible from cameras
        N_CAMERAS = 6,  # Number of cameras
    ),
    
    LIFT = SimpleNamespace(
        # Short BEV dimensions
        X_BOUND = [-15.0, 15.0, 0.15],  # Forward
        Y_BOUND = [-15.0, 15.0, 0.15],  # Sides
        Z_BOUND = [-10.0, 10.0, 20.0],  # Height
        D_BOUND = [2.0, 50.0, 1.0],
        
        # Long BEV dimensions
        # X_BOUND = [-50.0, 50.0, 0.5],  # Forward
        # Y_BOUND = [-50.0, 50.0, 0.5],  # Sides
        # Z_BOUND = [-10.0, 10.0, 20.0],  # Height
        # D_BOUND = [2.0, 50.0, 1.0],
    ),
    
    MODEL = SimpleNamespace(
    
        STCONV = SimpleNamespace(
            INPUT_EGOPOSE = True,
        ),
        
        SEMANTIC_HEAD = SimpleNamespace(
            BEV_DIM = 128,
            PAST_FRAMES = 3,
            TEXT_DIM = 512,
            HIDDEN_DIM = 1024,
            N_HIDDEN_LAYERS = 2,
            DROPOUT = 0.1,
        ),
        
        ENCODER = SimpleNamespace(
            DOWNSAMPLE = 8,
            NAME = 'efficientnet-b4',
            OUT_CHANNELS = 64,
            USE_DEPTH_DISTRIBUTION = True,
        ),

        # Tiny
        # SEGFORMER = SimpleNamespace(
        #     N_ENCODER_BLOCKS = 5,
        #     DEPTHS = [2, 2, 2, 2, 2],
        #     SEQUENCE_REDUCTION_RATIOS = [8, 4, 2, 1, 1],
        #     HIDDEN_SIZES = [16, 24, 32, 48, 64], 
        #     PATCH_SIZES = [7, 3, 3, 3, 3],
        #     STRIDES = [2, 2, 2, 2, 2],
        #     NUM_ATTENTION_HEADS = [1, 2, 4, 8, 8],
        #     MLP_RATIOS = [4, 4, 4, 4, 4],
        #     HEAD_DIM_MULTIPLIER = 4,
        #     HEAD_KERNEL = 2,
        #     HEAD_STRIDE = 2,
        # ),
                
        # B0
        SEGFORMER = SimpleNamespace(
            N_ENCODER_BLOCKS = 5,
            DEPTHS = [2, 2, 2, 2, 2],
            SEQUENCE_REDUCTION_RATIOS = [8, 8, 4, 2, 1],
            HIDDEN_SIZES = [16, 32, 64, 160, 256],
            PATCH_SIZES = [7, 3, 3, 3, 3],
            STRIDES = [2, 2, 2, 2, 2],
            NUM_ATTENTION_HEADS = [1, 1, 2, 4, 8],
            MLP_RATIOS = [4, 4, 4, 4, 4],
            HEAD_DIM_MULTIPLIER = 4,
            HEAD_KERNEL = 2,
            HEAD_STRIDE = 2,
        ),

        TEMPORAL_MODEL = SimpleNamespace(
            NAME = 'temporal_block',
            START_OUT_CHANNELS = 64,
            EXTRA_IN_CHANNELS = 0,
            INBETWEEN_LAYERS = 0,
            PYRAMID_POOLING = True,
            INPUT_EGOPOSE = True,
        ),
        DISTRIBUTION = SimpleNamespace(
            LATENT_DIM = 32,
            MIN_LOG_SIGMA = -5.0,
            MAX_LOG_SIGMA = 5.0,
        ),
        FUTURE_PRED = SimpleNamespace(
            N_GRU_BLOCKS = 3,
            N_RES_LAYERS = 3,
        ),

        BN_MOMENTUM = 0.1,
    ),
    
    # how many frames of temporal context (1 for single timeframe)
    TIME_RECEPTIVE_FIELD = 3, 
    # how many time steps into the future to predict
    N_FUTURE_FRAMES = 4,  

    IMAGE = SimpleNamespace(
        FINAL_DIM = (224, 480),
        ORIGINAL_DIM = (900, 1600),
        RESIZE_SCALE = 0.3,
        TOP_CROP = 46,
        ORIGINAL_HEIGHT = 900 ,  # Original input RGB camera height
        ORIGINAL_WIDTH = 1600 ,  # Original input RGB camera width
        NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    ),
    
    # IMAGE = SimpleNamespace(
    #     FINAL_DIM = (448, 800), # (224, 480),
    #     ORIGINAL_DIM = (900, 1600),
    #     RESIZE_SCALE = 0.5, # 0.3,
    #     TOP_CROP = 2, # 46,
    #     ORIGINAL_HEIGHT = 900 ,  # Original input RGB camera height
    #     ORIGINAL_WIDTH = 1600 ,  # Original input RGB camera width
    #     NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    #              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    # ),


    SEMANTIC_SEG = SimpleNamespace(
        # per class cross entropy weights (bg, dynamic, drivable, lane)
        WEIGHTS = [1.0, 2.0],
        USE_TOP_K = True,  # backprop only top-k hardest pixels
        TOP_K_RATIO = 0.25,
    ),

    INSTANCE_SEG = SimpleNamespace(),

    INSTANCE_FLOW = SimpleNamespace(
        ENABLED = True,
    ),

    PROBABILISTIC = SimpleNamespace(
        ENABLED = False,  # learn a distribution over futures
        WEIGHT = 100.0,
        # number of dimension added (future flow, future centerness, offset, seg)
        FUTURE_DIM = 6,
    ),

    FUTURE_DISCOUNT = 0.95,

    VISUALIZATION = SimpleNamespace(
        OUTPUT_PATH = './visualization_outputs',
        SAMPLE_NUMBER = 1000,
        VIS_GT = True,
    )
)

    
class FullSegformerCustomHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.feature_width = int(
            (self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0])
            / self.cfg.LIFT.X_BOUND[2]
        )

        self.use_ego_motion = self.cfg.MODEL.STCONV.INPUT_EGOPOSE
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        self.use_depth_distribution = self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION
        self.temporal_attn_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        if self.use_ego_motion:
            self.temporal_attn_channels += 6
        self.lidar_supervision = self.cfg.LIDAR_SUPERVISION

        self.feature_extractor = FeatureExtractor(
            x_bound=self.cfg.LIFT.X_BOUND,
            y_bound=self.cfg.LIFT.Y_BOUND,
            z_bound=self.cfg.LIFT.Z_BOUND,
            d_bound=self.cfg.LIFT.D_BOUND,
            downsample=self.cfg.MODEL.ENCODER.DOWNSAMPLE,
            out_channels=self.cfg.MODEL.ENCODER.OUT_CHANNELS,
            receptive_field=self.cfg.TIME_RECEPTIVE_FIELD,
            pred_frames=self.cfg.N_FUTURE_FRAMES,
            latent_dim=self.cfg.MODEL.DISTRIBUTION.LATENT_DIM,
            use_depth_distribution=self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION,
            model_name=self.cfg.MODEL.ENCODER.NAME,
            img_size=self.cfg.IMAGE.FINAL_DIM,
            return_depth_map=self.lidar_supervision,
        )

        segformer_out_dim = (
            len(self.cfg.SEMANTIC_SEG.WEIGHTS)
            * (self.cfg.N_FUTURE_FRAMES + 2)
            * self.cfg.MODEL.SEGFORMER.HEAD_DIM_MULTIPLIER
        )

        segformer_config = SegformerConfig(
            image_size=self.feature_width,
            num_channels=self.cfg.TIME_RECEPTIVE_FIELD * self.temporal_attn_channels,
            num_encoder_blocks=self.cfg.MODEL.SEGFORMER.N_ENCODER_BLOCKS,
            depths=self.cfg.MODEL.SEGFORMER.DEPTHS,
            sr_ratios=self.cfg.MODEL.SEGFORMER.SEQUENCE_REDUCTION_RATIOS,
            hidden_sizes=self.cfg.MODEL.SEGFORMER.HIDDEN_SIZES,  # No receptive field multiplication
            patch_sizes=self.cfg.MODEL.SEGFORMER.PATCH_SIZES,
            strides=self.cfg.MODEL.SEGFORMER.STRIDES,
            num_attention_heads=self.cfg.MODEL.SEGFORMER.NUM_ATTENTION_HEADS,
            mlp_ratios=self.cfg.MODEL.SEGFORMER.MLP_RATIOS,
            output_hidden_states=True,
            return_dict=True,
            num_labels=segformer_out_dim,
        )

        kernel = self.cfg.MODEL.SEGFORMER.HEAD_KERNEL
        stride = self.cfg.MODEL.SEGFORMER.HEAD_STRIDE
        segformer_out_dim = 256

        # Instantiate the two different banches and change the classifier layer with our head
        self.segmentation_branch = SegformerForSemanticSegmentation(segformer_config)
        self.segmentation_branch.decode_head.classifier = nn.Sequential(
            Residual(segformer_out_dim, segformer_out_dim // 2),
            Residual(segformer_out_dim // 2, segformer_out_dim // 2),
            Residual(segformer_out_dim // 2, segformer_out_dim // 4),
            Residual(segformer_out_dim // 4, segformer_out_dim // 4),
            nn.ConvTranspose2d(
                segformer_out_dim // 4,
                len(self.cfg.SEMANTIC_SEG.WEIGHTS) * (self.cfg.N_FUTURE_FRAMES + 2),
                kernel_size=kernel,
                stride=stride,
            ),
        )

        # segformer_config.num_labels = 2
        self.flow_branch = SegformerForSemanticSegmentation(segformer_config)
        self.flow_branch.decode_head.classifier = nn.Sequential(
            Residual(segformer_out_dim, segformer_out_dim // 2),
            Residual(segformer_out_dim // 2, segformer_out_dim // 2),
            Residual(segformer_out_dim // 2, segformer_out_dim // 4),
            Residual(segformer_out_dim // 4, segformer_out_dim // 4),
            nn.ConvTranspose2d(
                segformer_out_dim // 4,
                2 * (self.cfg.N_FUTURE_FRAMES + 2),
                kernel_size=kernel,
                stride=stride,
            ),
        )

        self.semantic_head = SemanticHead(
            bev_dim=self.cfg.MODEL.ENCODER.OUT_CHANNELS + 6,
            past_frames=self.cfg.TIME_RECEPTIVE_FIELD,
            text_dim=self.cfg.MODEL.SEMANTIC_HEAD.TEXT_DIM,
            hidden_dim=self.cfg.MODEL.SEMANTIC_HEAD.HIDDEN_DIM,
            n_hidden_layers=self.cfg.MODEL.SEMANTIC_HEAD.N_HIDDEN_LAYERS,
            dropout=self.cfg.MODEL.SEMANTIC_HEAD.DROPOUT,
        )

    def forward(
        self,
        imgs,
        intrins,
        rots,
        trans,
        future_egomotion,
        **kwargs,
    ):
        output = {}
        start_time = time.time()
        
        # Construct extrinsics from rots and trans
        extrinsics = torch.zeros(
            (*rots.shape[:3], 4, 4), dtype=rots.dtype, device=rots.device
        )
        
        extrinsics[..., :3, :3] = rots
        extrinsics[..., :3, 3:4] = trans
        extrinsics[..., 3, 3] = 1.0
        
        # Image feature extraction
        x, depth_maps = self.feature_extractor(
            imgs,                  # (x, sweeps, n_cam, channel, h, w)
            intrins,         # (b, sweeps, n_cam, 3, 3)
            extrinsics,         # (b, sweeps, n_cam, 4, 4)
            future_egomotion    # (b, sweeps, 6)
        )

        if self.lidar_supervision:
            output["depth_maps"] = depth_maps

        perception_time = time.time()

        # Transofrmer multi-scale encoder
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(
            b, s, c, h, w
        )
        # At time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat(
            [
                torch.zeros_like(future_egomotions_spatial[:, :1]),
                future_egomotions_spatial[:, : (self.receptive_field - 1)],
            ],
            dim=1,
        )
        x = torch.cat([x, future_egomotions_spatial], dim=-3)
        
        # Aux semantic head
        semantic_out = self.semantic_head(x)

        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)

        # Segformer directly
        seg_out = self.segmentation_branch(x).logits
        flow_out = self.flow_branch(x).logits
        
        output["binimg"] = seg_out.view(
            b, self.cfg.N_FUTURE_FRAMES + 2, len(self.cfg.SEMANTIC_SEG.WEIGHTS), h, w
        ).contiguous()
        output["flow"] = flow_out.view(
            b, self.cfg.N_FUTURE_FRAMES + 2, len(self.cfg.SEMANTIC_SEG.WEIGHTS), h, w
        ).contiguous()

        prediction_time = time.time()

        output["perception_time"] = perception_time - start_time
        output["prediction_time"] = prediction_time - perception_time
        output["total_time"] = output["perception_time"] + output["prediction_time"]

        output = {**output}

        return {"bev": output, "semantic": {"semantic_bev": semantic_out}}


## Feature extractor

def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])

class FeatureExtractor(nn.Module):
    def __init__(self,
                 x_bound = (-50.0, 50.0, 0.5),  # Forward
                 y_bound = (-50.0, 50.0, 0.5),  # Sides
                 z_bound = (-10.0, 10.0, 20.0),  # Height
                 d_bound = (2.0, 50.0, 1.0),
                 downsample: int = 8,
                 out_channels: int = 64,
                 receptive_field: int = 3,
                 pred_frames: int = 4,
                 latent_dim: int = 32,
                 use_depth_distribution: bool = True,
                 model_name: str = 'efficientnet-b0',
                 img_size = (224,480),
                 return_depth_map: bool = False,
                 ):
        super().__init__()

        self.bounds = {
            'x': x_bound,
            'y': y_bound,
            'z': z_bound,
            'd': d_bound
        }

        bev_resolution, bev_start_position, bev_dimension = (
            calculate_birds_eye_view_parameters(x_bound, y_bound, z_bound)
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.img_final_dim = img_size

        self.encoder_downsample = downsample
        self.encoder_out_channels = out_channels

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape
        self.return_depth_map = return_depth_map

        # temporal block
        self.receptive_field = receptive_field
        self.n_future = pred_frames
        # self.latent_dim = latent_dim

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (x_bound[1], y_bound[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Define the camera multi-sweep encoder
        if 'efficientnet' in model_name.lower():
            self.encoder = EncoderEfficientNet(
                out_channels=out_channels,
                depth_distribution=use_depth_distribution,
                depth_channels=self.depth_channels,
                downsample=downsample,
                model_name=model_name,
                return_depth_map=self.return_depth_map,
            )
        
        else:
            raise ValueError(f'Encoder model {model_name} not handled.')


    def forward(self, image, intrinsics, extrinsics, future_egomotion):
        '''
        Inputs:
            image: (b, sweeps, n_cam, channel, h, w)
            intrinsic: (b, sweeps, n_cam, 3, 3)
            extrinsic: (b, sweeps, n_cam, 4, 4)
            future_egomotion: (b, sweeps, 6)
        '''
        # Only process features from the past and present (within receptive field)
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()

        x, depth_maps = self.bev_features(image, extrinsics, intrinsics)

        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        return x, depth_maps

    def create_frustum(self) -> nn.Parameter:
        # Create grid in image plane
        h, w = self.img_final_dim
        downsampled_h, downsampled_w = (
            h // self.encoder_downsample,
            w // self.encoder_downsample,
        )

        # Depth grid
        depth_grid = torch.arange(*self.bounds["d"], dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(
            n_depth_slices, downsampled_h, downsampled_w
        )
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(
            n_depth_slices, downsampled_h, downsampled_w
        )

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def bev_features(self, x, extrinsics, intrinsics):
        '''
        Inputs:
            x: (b, sweeps, n_cam, channel, h, w)
            extrinsics: (b, sweeps, n_cam, 4, 4)
            intrinsics: (b, sweeps, n_cam, 3, 3)
        '''
        batch, sweeps, n_cam, channel, h, w = x.shape

        # Reshape to (b*sweeps*n_cam, channel, h, w)
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics) 

        geometry = self.get_geometry(intrinsics, extrinsics)
        x, depth = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = unpack_sequence_dim(x, batch, sweeps)
        if self.return_depth_map and depth is not None:
            depth = unpack_sequence_dim(depth, batch, sweeps)
        return x, depth

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features."""
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = (
            combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        )
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x, depth = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        if self.return_depth_map and depth is not None:
            depth = depth.view(b, n, *depth.shape[1:])

        return x, depth

    def projection_to_birds_eye_view(self, x, geometry):
        """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]),
            dtype=torch.float,
            device=x.device,
        )
        
        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = (
                geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)
            ) / self.bev_resolution
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                (geometry_b[:, 0] >= 0)
                & (geometry_b[:, 0] < self.bev_dimension[0])
                & (geometry_b[:, 1] >= 0)
                & (geometry_b[:, 1] < self.bev_dimension[1])
                & (geometry_b[:, 2] >= 0)
                & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                + geometry_b[:, 1] * (self.bev_dimension[2])
                + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = (
                x_b[ranks_indices],
                geometry_b[ranks_indices],
                ranks[ranks_indices],
            )

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros(
                (
                    self.bev_dimension[2],
                    self.bev_dimension[0],
                    self.bev_dimension[1],
                    c,
                ),
                device=x_b.device,
            )
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output
    
class EncoderEfficientNet(nn.Module):
    def __init__(self, 
                 out_channels: int = 64,
                 depth_distribution: bool = True,
                 pretrained: bool = True,
                 depth_channels: int = 32,
                 downsample: int = 8,
                 model_name: str = 'efficientnet-b0',
                 return_depth_map: bool = False,
                 ):
        super().__init__()
        self.D = depth_channels
        self.C = out_channels
        self.use_depth_distribution = depth_distribution
        self.downsample = downsample
        self.version = model_name.split("-")[1]
        self.return_depth_map = return_depth_map

        self.backbone = EfficientNet.from_pretrained(model_name)
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == "b0":
                upsampling_in_channels = 320 + 112
            elif self.version == "b4":
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == "b0":
                upsampling_in_channels = 112 + 40
            elif self.version == "b4":
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f"Downsample factor {self.downsample} not handled.")

        self.upsampling_layer = UpsamplingConcat(
            upsampling_in_channels, upsampling_out_channels
        )
        if self.use_depth_distribution:
            self.depth_layer = nn.Conv2d(
                upsampling_out_channels, self.C + self.D, kernel_size=1, padding=0
            )
        else:
            self.depth_layer = nn.Conv2d(
                upsampling_out_channels, self.C, kernel_size=1, padding=0
            )

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']

        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
                
        x = self.get_features(x)  # get feature vector

        x = self.depth_layer(x)  # feature and depth head
        
        if self.use_depth_distribution:
            depth = x[:, : self.D].softmax(dim=1)
            x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)  # outer product depth and features
            if self.return_depth_map:
                return x, depth
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return x, None
    
## Layers

class Residual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self._downsample = downsample
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        if upsample:
            assert (
                not downsample
            ), "downsample and upsample not possible simultaneously."
            conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
            )
        elif downsample:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
            )
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
            )

        self.layers = nn.Sequential(
            conv, nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True)
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update(
                    {"upsample_skip_proj": nn.Upsample(scale_factor=2, mode="bilinear")}
                )
            elif downsample:
                projection.update(
                    {"upsample_skip_proj": nn.MaxPool2d(kernel_size=2, stride=2)}
                )
            projection.update(
                {
                    "conv_skip_proj": nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, bias=False
                    ),
                    "bn_skip_proj": nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(
                    x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0
                )
            return x_residual + self.projection(x)
        return x_residual + x


class ConvBlock(nn.Module):
    """2D convolution followed by
         - an optional normalisation (batch norm or instance norm)
         - an optional activation (ReLU, LeakyReLU, or tanh)
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        norm='bn',
        activation='relu',
        bias=False,
        transpose=False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        # Define the main conv operation
        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=1,
                stride=2,
                output_padding=padding_size,
                padding=padding_size,
                groups=groups,
            )
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                stride=2,
                padding=padding_size,
                groups=groups,
            )
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                padding=padding_size,
                groups=groups,
            )

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # First projection with 1x1 kernel
                    ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
                    ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                                                       nn.ReLU(inplace=True))),
                    # Second conv block
                    ('conv', bottleneck_conv),
                    ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
                    ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
                                                     nn.ReLU(inplace=True))),
                    # Regulariser
                    ('dropout', nn.Dropout2d(p=dropout)),
                ]
            )
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    # pylint: disable=arguments-differ
    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    # pylint: disable=arguments-differ
    def forward(self, x):
        return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip    

## Utils

def namespace_to_dict(namespace):
    if isinstance(namespace, SimpleNamespace):
        result = {}
        for key, value in namespace.__dict__.items():
            result[key] = namespace_to_dict(value)
        return result
    elif isinstance(namespace, list):
        return [namespace_to_dict(item) for item in namespace]
    else:
        return namespace

def resize_and_crop_image(img, resize_dims, crop):
    # Bilinear resizing followed by cropping
    img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    """
    Parameters
    ----------
        intrinsics: torch.Tensor (3, 3)
        top_crop: float
        left_crop: float
        scale_width: float
        scale_height: float
    """
    updated_intrinsics = intrinsics.clone()
    # Adjust intrinsics scale due to resizing
    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    # Adjust principal point due to cropping
    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def invert_pose_matrix(x):
    """
    Parameters
    ----------
        x: [B, 4, 4] batch of pose matrices

    Returns
    -------
        out: [B, 4, 4] batch of inverse pose matrices
    """
    assert len(x.shape) == 3 and x.shape[1:] == (4, 4), 'Only works for batch of pose matrices.'

    transposed_rotation = torch.transpose(x[:, :3, :3], 1, 2)
    translation = x[:, :3, 3:]

    inverse_mat = torch.cat([transposed_rotation, -torch.bmm(transposed_rotation, translation)], dim=-1) # [B,3,4]
    inverse_mat = torch.nn.functional.pad(inverse_mat, [0, 0, 0, 1], value=0)  # [B,4,4]
    inverse_mat[..., 3, 3] = 1.0
    return inverse_mat


def warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Applies a rotation and translation to feature map x.
        Args:
            x: (b, c, h, w) feature map
            flow: (b, 6) 6DoF vector (only uses the xy poriton)
            mode: use 'nearest' when dealing with categorical inputs
        Returns:
            in plane transformed feature map
        """
    if flow is None:
        return x
    b, c, h, w = x.shape
    # z-rotation
    angle = flow[:, 5].clone()  # torch.atan2(flow[:, 1, 0], flow[:, 0, 0])
    # x-y translation
    translation = flow[:, :2].clone()  # flow[:, :2, 3]

    # Normalise translation. Need to divide by how many meters is half of the image.
    # because translation of 1.0 correspond to translation of half of the image.
    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]
    # forward axis is inverted
    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    # output = Rot.input + translation
    # tx and ty are inverted as is the case when going from real coordinates to numpy coordinates
    # translation_pos_0 -> positive value makes the image move to the left
    # translation_pos_1 -> positive value makes the image move to the top
    # Angle -> positive value in rad makes the image move in the trigonometric way
    transformation = torch.stack([cos_theta, -sin_theta, translation[:, 1],
                                  sin_theta, cos_theta, translation[:, 0]], dim=-1).view(b, 2, 3)

    # Note that a rotation will preserve distances only if height = width. Otherwise there's
    # resizing going on. e.g. rotation of pi/2 of a 100x200 image will make what's in the center of the image
    # elongated.
    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    warped_x = torch.nn.functional.grid_sample(x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def cumulative_warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.

    x[:, -1] remains unchanged
    x[:, -2] is warped using flow[:, -2]
    x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
    ...
    x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)

    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    flow = pose_vec2mat(flow)

    out = [x[:, -1]]
    cum_flow = flow[:, -2]
    for t in reversed(range(sequence_length - 1)):
        out.append(warp_features(x[:, t], mat2pose_vec(cum_flow), mode=mode, spatial_extent=spatial_extent))
        # @ is the equivalent of torch.bmm
        cum_flow = flow[:, t - 1] @ cum_flow

    return torch.stack(out[::-1], 1)


def cumulative_warp_features_reverse(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.

    x[:, 0] remains unchanged
    x[:, 1] is warped using flow[:, 0].inverse()
    x[:, 2] is warped using flow[:, 0].inverse() @ flow[:, 1].inverse()
    ...

    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)

    """
    flow = pose_vec2mat(flow)

    out = [x[:,0]]
    
    for i in range(1, x.shape[1]):
        if i==1:
            cum_flow = invert_pose_matrix(flow[:, 0])
        else:
            cum_flow = cum_flow @ invert_pose_matrix(flow[:,i-1])
        out.append( warp_features(x[:,i], mat2pose_vec(cum_flow), mode, spatial_extent=spatial_extent))
    return torch.stack(out, 1)


def flow_warp(occupancy, flow, mode='nearest', padding_mode='zeros'):
    """Warps ground-truth flow-origin occupancies according to predicted flows.

    Performs bilinear interpolation and samples from 4 pixels for each flow
    vector.

    Args:
      occupancy: occupancy as float32 tensors with the shape of BxTx1xHxW
      flow: flow as float32 tensors with the shape of BxTx2xHxW
      mode: mode of grid sample

    Returns:
      warped_occupancy: occupancy grids for vehicles as float32 tensors with the shape of BxTx1xHxW.

    Note: the flow must always be 1 timestep ahead of the corresponding occupancy
    """
    _, num_waypoints, _, grid_height_cells, grid_width_cells = occupancy.size()

    h = torch.linspace(-1, 1, steps=grid_height_cells)
    w = torch.linspace(-1, 1, steps=grid_width_cells)
    h_idx, w_idx = torch.meshgrid(h, w, indexing='ij')
    # These indices map each (x, y) location to the pixel (x, y).
    identity_indices = torch.stack((w_idx, h_idx), dim=0).to(device=occupancy.device)  # 2xHxW, storing x, y coordinates.

    warped_occupancy = []
    for k in range(num_waypoints):
        flow_origin_occupancy = occupancy[:, k]  # BxTx1xHxW -> Bx1xHxW
        pred_flow = flow[:, k]  # BxTx2xHxW -> Bx2xHxW
        # Normalize along the width and height direction
        normalize_pred_flow = torch.stack(
            (2.0 * pred_flow[:, 0] / (grid_width_cells - 1),  
            2.0 * pred_flow[:, 1] / (grid_height_cells - 1)),
            dim=1,
        )
        # Shift the identity grid indices according to predicted flow tells us
        # the source (origin) grid cell for each flow vector. We simply sample
        # occupancy values from these locations.
        warped_indices = identity_indices + normalize_pred_flow  # Bx2xHxW
        warped_indices = warped_indices.permute(0, 2, 3, 1)  # Bx2xHxW -> BxHxWx2
        sampled_occupancy = F.grid_sample(
            input=flow_origin_occupancy,
            grid=warped_indices,
            mode=mode,
            padding_mode='zeros',
            align_corners=True,
        )
        warped_occupancy.append(sampled_occupancy)
    warped_occupancy = torch.stack(warped_occupancy, dim=1)
    return warped_occupancy

class VoxelsSumming(torch.autograd.Function):
    """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193"""
    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None
    
## Trainer ##

class TrainingModule(L.LightningModule):
    def __init__(self, hparams, cfg):
        super().__init__()

        self.params = hparams
        self.save_hyperparameters()
        self.cfg = cfg

        self.n_classes = len(self.cfg.SEMANTIC_SEG.WEIGHTS)


        self.model = FullSegformerCustomHead(self.cfg)
        self.lidar_supervision = False
        
        # Uncertainty weighting
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0),
                                                      requires_grad=True)
        self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        
        self.training_step_count = 0

        # Run time
        self.perception_time, self.prediction_time, self.postprocessing_time = [], [], []
        

