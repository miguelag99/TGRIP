import torch

from torch import nn
from copy import deepcopy
from typing import List

from tgrip.utils.debug import debug_hook
from tgrip.models.layers import STconv, ConvNormAct


class MultiPyramidHead(nn.Module):
    def __init__(
        self,
        shared_out_c: int = 128,
        latent_dim: int = 64,
        in_seq_len: int = 3,
        out_seq_len: int = 6,
        # Outputs
        with_centr_offs: bool = False,
        with_hdmap: bool = False,
        hdmap_names: List[str] = [],
        with_binimg: bool = True,
        with_multiclass: bool = False,
        class_weights: List[float] = [],
        with_flow: bool = False,
    ):
        super().__init__()
        # assert with_flow and with_binimg, "Flow and binimg are required"
        self.register_forward_hook(debug_hook)
        dict_input = dict(
            with_binimg=with_binimg,
            with_centr_offs=with_centr_offs,
            with_hdmap=with_hdmap,
            with_flow=with_flow,
        )
        self.with_centr_offs = with_centr_offs
        self.with_flow = with_flow
        self.out_seq_len = out_seq_len
        self.in_seq_len = in_seq_len
        self.hdmap_names = hdmap_names
        self.with_multiclass = with_multiclass
        self.class_weights = class_weights
        self._get_layers(dict_input, shared_out_c, latent_dim)

    def _get_layers(self, dict_input, shared_out_c, latent_dim):
        # Unpack
        (with_binimg, with_centr_offs, with_hdmap, with_flow) = (
            dict_input["with_binimg"],
            dict_input["with_centr_offs"],
            dict_input["with_hdmap"],
            dict_input["with_flow"],
        )

        # Prepare out
        map_out = nn.ModuleDict()

        # Initialize layers.
        shared_in_c = shared_out_c * self.in_seq_len
        out_classes = len(self.class_weights) if self.with_multiclass else 1

        norm = nn.InstanceNorm2d
        
        # Segmentation head
        convnormact_conv_seg = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            # Rearrange from (b, t, c, h, w) to (b*t, c, h, w)
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=out_classes,
                        kernel_size=1, padding=0)
        )
        
        # Flow head
        convnormact_conv_flow = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            # Rearrange from (b, t, c, h, w) to (b*t, c, h, w)
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=2,
                        kernel_size=1, padding=0),
        )
        
        # Center and offset head
        convnormact_conv_offs = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=2,
                        kernel_size=1, padding=0),
        )
        convnormact_conv_centerness = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=1,
                        kernel_size=1, padding=0)
        )
        
        # HDMap head
        # if with_hdmap:
        #     convnormact_conv_chdmap = nn.Sequential(
        #         nn.Flatten(1, 2),
        #         ConvNormAct(shared_out_c*self.in_seq_len, latent_dim,
        #                     3, 1, False, norm, nn.ReLU),
        #         nn.Conv2d(
        #             latent_dim, len(self.hdmap_names),
        #             kernel_size=1, padding=0
        #         ),
        #     )
        
        # Initialize heads.
        map_out.update(
            {
                "binimg": deepcopy(convnormact_conv_seg),
                "offsets": deepcopy(convnormact_conv_offs),
                "centerness": deepcopy(convnormact_conv_centerness),
                "flow": deepcopy(convnormact_conv_flow),
            }
        )
        
        # Extra heads
        # if with_hdmap:
        #     map_out.update({"hdmap": convnormact_conv_chdmap})
        self.map_out = map_out

    def forward_layers(self, x):
        out_dict = {}
        out_dict.update({k: (layer(x)) for k, layer in self.map_out.items()})
        return out_dict

    def _apply_final_activation(self, out_dict):
        feats = out_dict["centerness"]
        out_dict["centerness"] = torch.sigmoid(feats)
        return out_dict

    def forward(self, x):
        b, t, c, h, w = x.shape
        out_dict = self.forward_layers(x)
        out_dict = self._apply_final_activation(out_dict)
        return out_dict


class SinglePyramidHead(nn.Module):
    def __init__(
        self,
        shared_out_c: int = 128,
        latent_dim: int = 128,
        in_seq_len: int = 3,
        out_seq_len: int = 6,
        # Outputs
        with_centr_offs: bool = False,
        with_hdmap: bool = False,
        hdmap_names: List[str] = [],
        with_binimg: bool = True,
        with_multiclass: bool = False,
        class_weights: List[float] = [],
        with_flow: bool = False,
    ):
        super().__init__()
        # assert with_flow and with_binimg, "Flow and binimg are required"
        self.register_forward_hook(debug_hook)
        dict_input = dict(
            with_binimg=with_binimg,
            with_centr_offs=with_centr_offs,
            with_hdmap=with_hdmap,
            with_flow=with_flow,
        )
        self.with_centr_offs = with_centr_offs
        self.with_flow = with_flow
        self.out_seq_len = out_seq_len
        self.in_seq_len = in_seq_len
        self.hdmap_names = hdmap_names
        self.with_multiclass = with_multiclass
        self.class_weights = class_weights
        self._get_layers(dict_input, shared_out_c, latent_dim)

    def _build_head(self, in_channels: int, out_channels: int, num_convs: int = 3):
        '''
        Helper function to build a deeper task-specific head.
        Stacks multiple 3x3 convs with Norm and ReLU before the final 1x1 projection.
        '''
        layers = []
        norm = nn.InstanceNorm2d
        
        # Build the refinement buffer (Task-specific feature extraction)
        for _ in range(num_convs):
            layers.append(
                ConvNormAct(in_channels, in_channels,
                            3, 1, False, norm, nn.ReLU)
            )
            
        # Final projection to the required output channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        
        return nn.Sequential(*layers)

    def _get_layers(self, dict_input, shared_out_c, latent_dim):
        # Unpack
        (with_binimg, with_centr_offs, with_hdmap, with_flow) = (
            dict_input["with_binimg"],
            dict_input["with_centr_offs"],
            dict_input["with_hdmap"],
            dict_input["with_flow"],
        )

        # Prepare out
        map_out = nn.ModuleDict()

        # Initialize layers.
        out_classes = len(self.class_weights) if self.with_multiclass else 1

        norm = nn.InstanceNorm2d
        
        self.common_predictor_pyramid =  nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[32, 48, 64, 96, 128], # Increased capacity
                n_blocks=3,
            ),
            # Rearrange from (b, t, c, h, w) to (b*t, c, h, w)
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
        )
        
        # Segmentation head
        if with_binimg:
            map_out["binimg"] = self._build_head(latent_dim, out_classes, num_convs=3)
                
        # Flow head
        if with_flow:
            map_out["flow"] = self._build_head(latent_dim, out_channels=2, num_convs=3)
                
        # Center and offset head
        if with_centr_offs:
            map_out["offsets"] = self._build_head(latent_dim, out_channels=2, num_convs=3)
            map_out["centerness"] = self._build_head(latent_dim, out_channels=1, num_convs=3)
                
        self.map_out = map_out

    def forward_layers(self, x):
        out_dict = {}
        x = self.common_predictor_pyramid(x)
        out_dict.update({k: (layer(x)) for k, layer in self.map_out.items()})
        return out_dict

    def _apply_final_activation(self, out_dict):
        if "centerness" in out_dict:
            out_dict["centerness"] = torch.sigmoid(out_dict["centerness"])
        return out_dict

    def forward(self, x):
        b, t, c, h, w = x.shape
        out_dict = self.forward_layers(x)
        out_dict = self._apply_final_activation(out_dict)
        return out_dict