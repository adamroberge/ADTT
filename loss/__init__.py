from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .ce_loss import CeLoss, CeWeightLoss, CeSelectLoss, FocalLoss
from .plan_reg_loss_lidar import PlanRegLossLidar
from .emb_loss import VQVAEEmbedLoss
from .recon_loss import ReconLoss, UnknownOccReconLoss, LovaszLoss, ImgLoss, PlaneLoss, PlaneWeightLoss, MaskedPlaneLoss
from .pose_loss import PoseLoss
from .sce_loss import SceLoss
from .temporal_loss import TemporalConsistencyLoss
from .graph_dyn_loss import GraphDynLoss
from .custom_focal_loss import CustomFocalLoss
from .center_loss import CenterLoss
from .control_token_loss import LearnableControlTokenLoss
