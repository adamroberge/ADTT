from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import torch


@OPENOCC_LOSS.register_module()
class CeLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = cls_weight

    def ce_loss(self, ce_inputs, ce_labels):

        weight = None
        if self.use_weight and self.cls_weight is not None:
            if isinstance(self.cls_weight, (list, tuple)):
                weight = torch.tensor(self.cls_weight, dtype=torch.float32, device=ce_inputs.device)
            elif isinstance(self.cls_weight, torch.Tensor):
                weight = self.cls_weight.to(ce_inputs.device)
            else:
                weight = None

        if len(ce_inputs.shape) == 4:
            assert ce_inputs.shape[0] == ce_labels.shape[0]
            ce_loss = F.cross_entropy(ce_inputs, ce_labels, weight=weight, ignore_index=self.ignore)
        else:
            assert ce_inputs.shape[0] == ce_labels.shape[0]
            assert ce_inputs.shape[1] == ce_labels.shape[1]
            num_classes = ce_inputs.shape[-1]
            ce_loss = F.cross_entropy(ce_inputs.view(-1, num_classes), ce_labels.view(-1,),
                                     weight=weight, ignore_index=self.ignore)
        return ce_loss

@OPENOCC_LOSS.register_module()
class CeWeightLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_weight_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None

    def ce_weight_loss(self, ce_inputs, ce_labels):
        time_weights = torch.tensor([1.0, 0.85, 0.7, 0.55, 0.4, 0.25],
                                device=ce_inputs.device, dtype=ce_inputs.dtype)

        if len(ce_inputs.shape) == 4:
            B, T, N, C = ce_inputs.shape
            losses = []
            for t in range(T):
                loss_t = F.cross_entropy(
                    ce_inputs[:, t].reshape(-1, C),
                    ce_labels[:, t].reshape(-1),
                    reduction='mean',
                    ignore_index=self.ignore,
                    weight=self.cls_weight if self.use_weight else None
                )
                losses.append(loss_t * time_weights[t])
            ce_loss = sum(losses) / time_weights[:T].sum()
        else:
            assert ce_inputs.shape[0] == ce_labels.shape[0]
            assert ce_inputs.shape[1] == ce_labels.shape[1]
            ce_loss = F.cross_entropy(ce_inputs.view(-1, 18), ce_labels.view(-1,))
        return ce_loss


@OPENOCC_LOSS.register_module()
class CeSelectLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None

    def ce_loss(self, ce_inputs, ce_labels):
        if len(ce_inputs.shape) == 4:
            assert ce_inputs.shape[0] == ce_labels.shape[0]
            B, T, N, C = ce_inputs.shape
            T_select = min(2, T)
            ce_loss = F.cross_entropy(ce_inputs[:, :T_select], ce_labels[:, :T_select])
        else:
            assert ce_inputs.shape[0] == ce_labels.shape[0]
            assert ce_inputs.shape[1] == ce_labels.shape[1]
            ce_loss = F.cross_entropy(ce_inputs.view(-1, 18), ce_labels.view(-1,))
        return ce_loss


@OPENOCC_LOSS.register_module()
class FocalLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100,
                 gamma=2.0, alpha=None, use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.focal_loss
        self.ignore = ignore_label
        self.gamma = gamma
        self.use_weight = use_weight
        self.alpha = alpha
        self.cls_weight = cls_weight

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, float):
                pass
            else:
                self.alpha = None

    def focal_loss(self, ce_inputs, ce_labels):
        if self.alpha is None:
            alpha = None
        elif isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(ce_inputs.device)
        elif isinstance(self.alpha, (list, tuple)):
            alpha = torch.tensor(self.alpha, dtype=torch.float32, device=ce_inputs.device)
        elif isinstance(self.alpha, float):
            alpha = float(self.alpha)
        else:
            alpha = None

        weight = None
        if self.use_weight and self.cls_weight is not None:
            if isinstance(self.cls_weight, (list, tuple)):
                weight = torch.tensor(self.cls_weight, dtype=torch.float32, device=ce_inputs.device)
            elif isinstance(self.cls_weight, torch.Tensor):
                weight = self.cls_weight.to(ce_inputs.device)
            else:
                weight = None

        if len(ce_inputs.shape) == 4:
            ce_loss = F.cross_entropy(ce_inputs, ce_labels, reduction='none', ignore_index=self.ignore, weight=weight)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss

            valid_mask = ce_labels != self.ignore
            if valid_mask.sum() > 0:
                focal_loss_valid = focal_loss[valid_mask]

                if alpha is not None:
                    num_classes = ce_inputs.shape[1]
                    if isinstance(alpha, float):
                        if num_classes != 2:
                            raise ValueError(f"alpha为float仅支持二分类，当前类别数={num_classes}")
                        alpha_vec = torch.tensor([1.0 - alpha, alpha], dtype=torch.float32, device=ce_inputs.device)
                    else:
                        alpha_vec = alpha

                    if alpha_vec.shape[0] != num_classes:
                        raise ValueError(f"alpha长度与类别数不匹配: {alpha_vec.shape[0]} vs {num_classes}")

                    alpha_per_sample = alpha_vec[ce_labels[valid_mask]]
                    focal_loss_valid = alpha_per_sample * focal_loss_valid

                focal_loss = focal_loss_valid.mean()
            else:
                focal_loss = focal_loss.mean()
        else:
            assert ce_inputs.shape[0] == ce_labels.shape[0]
            assert ce_inputs.shape[1] == ce_labels.shape[1]
            num_classes = ce_inputs.shape[-1]

            ce_loss = F.cross_entropy(ce_inputs.view(-1, num_classes), ce_labels.view(-1,),
                                     reduction='none', ignore_index=self.ignore, weight=weight)

            pt = torch.exp(-ce_loss)

            focal_loss = (1 - pt) ** self.gamma * ce_loss

            flat_labels = ce_labels.view(-1)
            valid_mask = flat_labels != self.ignore
            if valid_mask.sum() > 0:
                focal_loss_valid = focal_loss[valid_mask]

                if alpha is not None:
                    if isinstance(alpha, float):
                        if num_classes != 2:
                            raise ValueError(f"alpha为float仅支持二分类，当前类别数={num_classes}")
                        alpha_vec = torch.tensor([1.0 - alpha, alpha], dtype=torch.float32, device=ce_inputs.device)
                    else:
                        alpha_vec = alpha

                    if alpha_vec.shape[0] != num_classes:
                        raise ValueError(f"alpha长度与类别数不匹配: {alpha_vec.shape[0]} vs {num_classes}")

                    alpha_per_sample = alpha_vec[flat_labels[valid_mask]]
                    focal_loss_valid = alpha_per_sample * focal_loss_valid

                focal_loss = focal_loss_valid.mean()
            else:
                focal_loss = focal_loss.mean()

        return focal_loss


@OPENOCC_LOSS.register_module()
class InpaintClearLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        clear_weight: float = 1.0,
        bg_weight: float = 1.0,
        ignore_label: int = -100,
        input_dict=None,
        **kwargs,
    ):
        super().__init__(weight)
        if input_dict is None:
            self.input_dict = {
                'clear_logits': 'inpaint_clear_logits',
                'clear_targets': 'inpaint_clear_targets',
                'bg_logits': 'inpaint_logits_bg',
                'bg_labels': 'inpaint_bg_labels',
            }
        else:
            self.input_dict = input_dict
        self.clear_weight = float(clear_weight)
        self.bg_weight = float(bg_weight)
        self.ignore = int(ignore_label)
        self.loss_func = self._loss

    def _loss(self, clear_logits, clear_targets, bg_logits, bg_labels):
        if clear_logits is None or clear_targets is None or bg_logits is None or bg_labels is None:
            return torch.zeros((), device=bg_logits.device if torch.is_tensor(bg_logits) else 'cpu', requires_grad=True)
        clear_logits = clear_logits.float()
        clear_targets = clear_targets.float()
        bg_logits = bg_logits.float()
        bg_labels = bg_labels.long()

        loss_clear = F.binary_cross_entropy_with_logits(clear_logits, clear_targets)
        num_classes = int(bg_logits.shape[-1])
        loss_bg = F.cross_entropy(bg_logits.reshape(-1, num_classes), bg_labels.reshape(-1), ignore_index=self.ignore)
        return self.clear_weight * loss_clear + self.bg_weight * loss_bg
