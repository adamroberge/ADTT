"""Paper-compatible adapters around the released representation and V21 model."""
import torch
from torch.utils.checkpoint import checkpoint

from model import TripLane, DeltaTokenPredictorV21
from model.TransDelta import quat_wxyz_to_rotmat, quat_wxyz_to_yaw, wrap_angle
from .config import LABELS


def representation_config(config):
    representation = config["representation"]
    return dict(encoder_cfg=dict(type="TripLaneEncoder", geo_feat_channels=representation["channels"],
                                 z_down=representation["z_down"], all_down=False),
                decoder_cfg=dict(type="TripLaneDecoder", in_channels=100,
                                 out_channels=len(LABELS[config["data"]["label_space"]]),
                                 hidden_channels=representation["decoder_hidden"],
                                 num_hidden_layers=representation["decoder_layers"], posenc=0),
                num_classes=len(LABELS[config["data"]["label_space"]]), expansion=representation["channels"])


def controls_from_poses(rotation, translation, history, future):
    """[dx,dy,delta_yaw] for the SAME (target-1)->target interval, Eq. 2.

    Translation is expressed in the previous ego frame, in metres per step;
    quaternions are wxyz. This avoids the legacy helper's silent zero-yaw fallback
    and its requirement for an extra pose beyond the final predicted frame.
    """
    target = slice(history, history + future)
    source = slice(history - 1, history + future - 1)
    matrices = quat_wxyz_to_rotmat(rotation)
    displacement = translation[target] - translation[source]
    local = torch.bmm(displacement[:, None], matrices[source]).squeeze(1)
    yaw = quat_wxyz_to_yaw(rotation)
    return torch.cat((local[:, :2], wrap_angle(yaw[target] - yaw[source])[:, None]), dim=-1)


class ForecastModel(DeltaTokenPredictorV21):
    def train(self, mode=True):
        super().train(mode)
        for module in (self.triplane_encoder, self.triplane_class_embeds, self.triplane_encoder_norm):
            module.eval()
        return self

    def _validated_poses(self, meta, total_steps, device, dtype):
        translation = torch.as_tensor(meta["ego2global_translation_seq"], device=device, dtype=dtype)
        rotation = torch.as_tensor(meta["ego2global_rotation_seq"], device=device, dtype=dtype)
        if translation.shape != (total_steps, 3) or rotation.shape != (total_steps, 4):
            raise ValueError("Complete historical and planned future poses are required")
        if not bool(torch.isfinite(translation).all() & torch.isfinite(rotation).all()):
            raise ValueError("Non-finite poses")
        if not torch.allclose(rotation.norm(dim=-1), torch.ones(total_steps, device=device, dtype=dtype), atol=1e-3):
            raise ValueError("Quaternions must be unit wxyz")
        return rotation, translation

    def _build_control_features(self, metas, future_steps, device):
        if not metas:
            raise ValueError("Pose metadata is required; identity/zero fallbacks are disabled")
        controls = []
        for meta in metas:
            rotation, translation = self._validated_poses(
                meta, self.prev_steps + future_steps, device, torch.float32)
            controls.append(controls_from_poses(rotation, translation, self.prev_steps, future_steps))
        return torch.stack(controls)

    def _get_ego_pose_sequence(self, metas, total_steps, device, dtype):
        if not metas:
            raise ValueError("Pose metadata is required")
        values = [self._validated_poses(meta, total_steps, device, dtype) for meta in metas]
        return torch.stack([quat_wxyz_to_rotmat(r) for r, _ in values]), torch.stack([t for _, t in values])


def build_model(config):
    representation = representation_config(config)
    if config["stage"] == "representation":
        return TripLane(**representation)
    forecast = config["forecast"]
    return ForecastModel(
        prev_steps=forecast["history"], num_frames=forecast["history"] + forecast["future"],
        plane_channels=config["representation"]["channels"], token_dim=forecast["token_dim"],
        context_frames=forecast["history"], transformer_depth=forecast["depth"],
        num_heads=forecast["heads"], dropout=forecast["dropout"],
        xy_hw=tuple(forecast["token_grid"][:2]),
        xz_hw=(forecast["token_grid"][0], forecast["token_grid"][2]),
        yz_hw=(forecast["token_grid"][1], forecast["token_grid"][2]),
        encoder_grid_size=tuple(forecast["token_grid"]), control_dim=3,
        num_real_control_tokens=forecast["ego_queries"],
        num_learnable_control_tokens=forecast["external_queries"],
        triplane_encoder_cfg=representation["encoder_cfg"],
        triplane_decoder_cfg=representation["decoder_cfg"], num_classes=representation["num_classes"],
        use_motion_compensation=True, predict_delta=True,
        warp_sample_mode="nearest", warp_padding_mode="zeros")


def transfer_representation(model, state):
    """Strict submodule transfer; never silently accept a partially loaded encoder."""
    pairs = {"encoder": "triplane_encoder", "class_embeds": "triplane_class_embeds",
             "norm": "triplane_encoder_norm", "geo_convs": "geo_convs", "decoder": "semantic_decoder"}
    for source, target in pairs.items():
        selected = {key[len(source) + 1:]: value for key, value in state.items() if key.startswith(source + ".")}
        getattr(model, target).load_state_dict(selected, strict=True)


def encode_sequence(model, labels):
    if isinstance(model, TripLane):
        planes = model.forward_encoder(labels)
        return [p.reshape(labels.shape[0], labels.shape[1], *p.shape[1:]) for p in planes]
    with torch.no_grad():
        return list(model._encode_triplanes_from_occs(labels))


def make_metas(batch_rows):
    return [dict(ego2global_translation_seq=[r["ego2global_translation"] for r in rows],
                 ego2global_rotation_seq=[r["ego2global_rotation"] for r in rows]) for rows in batch_rows]


def forecast(model, observed, metas, future_steps, on_step=None, step_embedding_period=None):
    if observed.shape[1] != model.prev_steps or len(metas) != observed.shape[0]:
        raise ValueError("Forecast input must contain exactly the observed history and matching poses")
    history = encode_sequence(model, observed)
    return model.forecast_latents(history, metas, future_steps,
                                 on_step=on_step, step_embedding_period=step_embedding_period)["predicted_planes"]


def query_logits(model, planes, queries, chunk_size, checkpoint_chunks=False):
    """Query chunks preserve the global CE/Lovasz objective; no dense logit grids."""
    def decode(xy, xz, yz, coordinates):
        if isinstance(model, TripLane):
            return model.forward_decoder([xy, xz, yz], coordinates)
        return model.semantic_decoder(model._triplane_query_feat([xy, xz, yz], coordinates))
    outputs = []
    for begin in range(0, queries.shape[1], chunk_size):
        coordinates = queries[:, begin:begin + chunk_size].contiguous()
        if checkpoint_chunks and torch.is_grad_enabled():
            outputs.append(checkpoint(decode, *planes, coordinates, use_reentrant=False))
        else:
            outputs.append(decode(*planes, coordinates))
    return torch.cat(outputs, dim=1)
