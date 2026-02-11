import os
import torch
import torch.nn as nn
from lightning.pytorch.core import LightningModule
from collections import defaultdict
from functools import partial

from tgrip.data.dataset.nuscenes_common import MAP_DYNAMIC_TAG, VISIBILITY_TAG
from tgrip.loss import BCELoss, CELoss, SpatialLoss, CosineSimilarityLoss, Weighting
from tgrip.metric import (
    IoUMetric,
    MeanMetric,
    IntersectionOverUnion,
    PanopticMetric,
    CosineSimilarityMetric,
)
from tgrip.utils import (
    GeomScaler, 
    nested_dict_to_nested_module_dict,
    get_ckpt_from_path,
    load_state_model,
    get_pylogger
)
from tgrip.models import PredictionTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = get_pylogger(__name__)

class DistillationPredictionTrainer(PredictionTrainer):
    def __init__(
        self,
        net,
        teacher,
        teacher_ckpt_path: str,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        weights_kwargs={},
        train_kwargs={},
        val_kwargs={},
        loss_kwargs={},
        metric_kwargs={},
        temporal_kwargs={},
        distill_kwargs={},
        grid={"xbound": [], "ybound": [], "zbound": []},
        text_encoder: nn.Module = None,
        name="",
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            weights_kwargs=weights_kwargs,
            train_kwargs=train_kwargs,
            val_kwargs=val_kwargs,
            loss_kwargs=loss_kwargs,
            metric_kwargs=metric_kwargs,
            temporal_kwargs=temporal_kwargs,
            grid=grid,
            text_encoder=text_encoder,
            name=name,
        )
        
        # Load frozen teacher
        self.teacher = teacher
        log.info(f"Loading teacher model from {teacher_ckpt_path}")

        teacher_ckpt = get_ckpt_from_path(teacher_ckpt_path)
        self.teacher = load_state_model(
            self.teacher, teacher_ckpt,
            keys_to_freeze='all', keys_to_load='all', verbose=1
        ).net.eval()
        log.info("Teacher model loaded")

        import pdb; pdb.set_trace()
        
        # TODO: finish setup
    
    def _init_loss(self, loss_kwargs):
        dict_losses = defaultdict(lambda: defaultdict(dict))

        cls_loss_segm = loss_kwargs.get("segm_type").get("cls")
        cls_loss_kwargs = loss_kwargs.get("segm_type").get("kwargs")
        loss_segm = partial(eval(cls_loss_segm), **cls_loss_kwargs)

        # -> BEV
        self.with_binimg = loss_kwargs.get("with_binimg", False)
        if self.with_binimg:
            if cls_loss_segm == "BCELoss":
                for index, elem in enumerate(self.bev_T_P):
                    # Filter by activated outputs
                    dict_losses["bev"]["binimg"][f"T{elem[0]}_P{elem[1]}"] = loss_segm(
                        time_index=index
                    )
            elif cls_loss_segm == "CELoss":
                dict_losses["bev"]["binimg"]["CELoss"] = loss_segm()
            else:
                raise NotImplementedError(f"{cls_loss_segm} not implemented.")
        
        # -> Centerness, offsets.
        self.with_centr_offs = loss_kwargs.get("with_centr_offs", False)
        if self.with_centr_offs:
            dict_losses["bev"].update(
                {
                    "centerness": SpatialLoss(norm=2),
                    "offsets": SpatialLoss(norm=1, ignore_index=255.0),
                }
            )
            
        # -> Flow loss
        self.with_flow = loss_kwargs.get("with_flow", False)
        if self.with_flow:
            dict_losses["bev"].update(
                {
                    "flow": SpatialLoss(norm=1.5, ignore_index=255.0),
                }
            )

        # -> Semantic map loss
        self.with_semantic_map = loss_kwargs.get("with_semantic_map", False)
        if self.with_semantic_map:
            dict_losses["bev"].update(
                {
                    "semantic_similarity": CosineSimilarityLoss(),
                }
            )
            
        # -> Teacher-Student distillation losses
        
        return dict_losses