"""Both training stages with deterministic batch cursors and dense validation."""
import json
import math
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import (atomic_json, load_checkpoint, restore_rng, rng_state,
                         save_checkpoint, source_receipt)
from .config import LABELS, manifest_path, resume_contract
from .data import OccupancyDataset, ResumableBatches, collate
from .losses import latent_loss, occupancy_loss
from .manifest import sha256
from .metrics import Confusion
from .models import (build_model, encode_sequence, forecast, make_metas,
                     query_logits, transfer_representation)
from .sampling import get_query
from .schedule import learning_rate_factor


def initialize_runtime(args, guard):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda":
        guard.check()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA device")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    return torch.device(args.device)


def sample_queries(labels, masks, config):
    """Sampling is in the main process so saved RNG state covers every draw."""
    count = len(LABELS[config["data"]["label_space"]])
    sampling = config["sampling"]
    queries, targets, selected_masks = [], [], []
    for frame, mask in zip(labels.flatten(0, 1), masks.flatten(0, 1)):
        q, target, centers = get_query(
            frame.numpy(), num_class=count, grid_size=tuple(frame.shape),
            max_points=sampling["points"], sampling_strategy=sampling["strategy"],
            empty_label=count - 1, boundary_distance=sampling["boundary_distance"],
            occupied_ratio=sampling["occupied_ratio"])
        if len(target) != sampling["points"]:
            raise ValueError("Sampler returned an unexpected query count")
        queries.append(q)
        targets.append(target)
        selected_masks.append(mask[centers[:, 0], centers[:, 1], centers[:, 2]])
    return torch.stack(queries), torch.stack(targets), torch.stack(selected_masks)


def compute_training_loss(model, batch, config, device, guard):
    history = config["forecast"]["history"]
    if config["stage"] == "representation":
        target_labels, target_masks = batch["labels"], batch["mask"]
        predictions = encode_sequence(model, batch["labels"].to(device))
        latent, components = None, {}
    else:
        target_labels, target_masks = batch["labels"][:, history:], batch["mask"][:, history:]
        observed = batch["labels"][:, :history].to(device)
        predictions = forecast(model, observed, make_metas(batch["rows"]),
                               config["forecast"]["future"], on_step=guard.step)
        # Future occupancy is encoded solely for a detached training target, never
        # passed into the autoregressive predictor or its history.
        target_planes = encode_sequence(model, target_labels.to(device))
        latent, components = latent_loss(predictions, target_planes, config["loss"])
    queries, targets, masks = sample_queries(target_labels, target_masks, config)
    logits = query_logits(model, [plane.flatten(0, 1) for plane in predictions],
                          queries.to(device), config["training"]["decode_chunk"],
                          config["training"]["checkpoint_decoder"])
    occupancy, occupancy_components = occupancy_loss(logits, targets.to(device), config["loss"], masks.to(device))
    components.update(occupancy_components)
    total = occupancy if latent is None else latent + config["loss"]["occupancy"] * occupancy
    components["total"] = total
    return total, components, torch.bincount(targets.flatten(), minlength=logits.shape[-1]).tolist()


def dense_prediction(model, planes, grid_size, chunk_size, device):
    """Visit every voxel; preserve the released endpoint query normalization."""
    nx, ny, nz = grid_size
    output = torch.empty(nx * ny * nz, dtype=torch.uint8)
    denominator = torch.tensor([nx - 1, ny - 1, nz - 1], device=device)
    for begin in range(0, output.numel(), chunk_size):
        end = min(begin + chunk_size, output.numel())
        flat = torch.arange(begin, end, device=device)
        centers = torch.stack((flat // (ny * nz), (flat // nz) % ny, flat % nz), dim=-1)
        queries = (2 * centers.float() / denominator - 1)[None]
        logits = query_logits(model, planes, queries, chunk_size)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite dense semantic predictions")
        output[begin:end] = logits.argmax(-1).reshape(-1).cpu().to(torch.uint8)
    return output.reshape(*grid_size)


@torch.no_grad()
def evaluate(model, dataset, config, device, guard, limit=None):
    was_training = model.training
    model.eval()
    names = LABELS[config["data"]["label_space"]]
    overall = Confusion(names, len(names) - 1)
    horizon_count = config["forecast"]["future"] if config["stage"] == "forecast" else 1
    per_horizon = [Confusion(names, len(names) - 1) for _ in range(horizon_count)]
    selected = dataset.validation_indices(config["training"]["validation_frames"] if limit is None else limit)
    input_receipts = []
    started = time.perf_counter()
    for index in selected:
        guard.step()
        batch = collate([dataset[index]])
        input_receipts.append(dict(index=index, tokens=[r["token"] for r in batch["rows"][0]],
                                   frame_content_sha256=batch["frame_content_sha256"][0]))
        if config["stage"] == "representation":
            planes = encode_sequence(model, batch["labels"].to(device))
            labels, masks = batch["labels"][0], batch["mask"][0]
        else:
            history = config["forecast"]["history"]
            planes = forecast(model, batch["labels"][:, :history].to(device),
                              make_metas(batch["rows"]), horizon_count, on_step=guard.step)
            labels, masks = batch["labels"][0, history:], batch["mask"][0, history:]
        for horizon in range(horizon_count):
            guard.step()
            prediction = dense_prediction(model, [p[:, horizon] for p in planes],
                config["data"]["grid_size"], config["training"]["decode_chunk"], device)
            overall.update(prediction, labels[horizon], masks[horizon])
            per_horizon[horizon].update(prediction, labels[horizon], masks[horizon])
    model.train(was_training)
    return dict(aggregate=overall.result(), per_prediction_step=[c.result() for c in per_horizon],
                validation_indices=selected, items=len(selected), frames=len(selected) * horizon_count,
                inputs=input_receipts,
                visibility=config["data"].get("visibility", "all"),
                trajectory="recorded poses; paired evaluation", seconds=time.perf_counter() - started)


def train(config, run, guard, args):
    device = initialize_runtime(args, guard)
    random.seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])
    train_path = manifest_path(config["data"]["train_manifest"])
    val_path = manifest_path(config["data"]["val_manifest"])
    config["data"]["train_manifest"] = str(train_path)
    config["data"]["val_manifest"] = str(val_path)
    training_data, validation_data = OccupancyDataset(train_path, config), OccupancyDataset(val_path, config)
    if {r["scene"] for r in training_data.rows} & {r["scene"] for r in validation_data.rows}:
        raise ValueError("Training and validation scenes overlap")
    if {r["token"] for r in training_data.rows} & {r["token"] for r in validation_data.rows}:
        raise ValueError("Training and validation tokens overlap")
    manifests = {"train": sha256(train_path), "val": sha256(val_path)}
    (run / "manifests").mkdir()
    shutil.copyfile(train_path, run / "manifests/train.jsonl")
    shutil.copyfile(val_path, run / "manifests/val.jsonl")
    atomic_json(run / "configuration.json", config)
    provenance = source_receipt(run)
    provenance.update(manifests=manifests, training_items=len(training_data), validation_items=len(validation_data),
                      training_scenes=sorted({r["scene"] for r in training_data.rows}),
                      validation_scenes=sorted({r["scene"] for r in validation_data.rows}),
                      device=str(device), gpu_uuid=args.gpu_uuid, memory_fraction=args.memory_fraction,
                      resume_checkpoint=str(Path(args.resume).resolve(strict=True)) if args.resume else None,
                      resume_checkpoint_sha256=sha256(args.resume) if args.resume else None,
                      representation_checkpoint=str(Path(args.representation_checkpoint).resolve(strict=True)) if args.representation_checkpoint else None,
                      determinism="Seeded RNG and resumable main-process sampling; CUDA kernels may be nondeterministic",
                      representation_checkpoint_sha256=sha256(args.representation_checkpoint) if args.representation_checkpoint else None)
    atomic_json(run / "provenance.json", provenance)
    guard.step()
    model = build_model(config).to(device)
    if config["stage"] == "forecast" and not args.resume:
        representation = load_checkpoint(args.representation_checkpoint)
        origin = representation["config"]
        if origin["stage"] != "representation" or origin["representation"] != config["representation"] or any(
                origin["data"][key] != config["data"][key] for key in ("label_space", "grid_size")):
            raise ValueError("Representation checkpoint class/geometry/model contract mismatch")
        if representation["manifests"] != manifests:
            raise ValueError("Representation pretraining must use the same approved train/val manifests")
        transfer_representation(model, representation["model"])
    training = config["training"]
    optimizer_type = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[training["optimizer"]]
    optimizer = optimizer_type([p for p in model.parameters() if p.requires_grad],
                               lr=training["lr"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, training))
    step, epoch, next_batch, best_score = 0, 0, 0, None
    if args.resume:
        saved = load_checkpoint(args.resume)
        if saved["contract"] != resume_contract(config) or saved["manifests"] != manifests:
            raise ValueError("Resume would change the mathematical recipe or dataset")
        if saved["provenance"]["training_source_sha256"] != provenance["training_source_sha256"]:
            raise ValueError("Training/model source changed; an exact resume requires the checkpoint's source version")
        if config["stage"] == "forecast":
            for key in ("representation_checkpoint", "representation_checkpoint_sha256"):
                provenance[key] = saved["provenance"][key]
            atomic_json(run / "provenance.json", provenance)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, epoch, next_batch, best_score = saved["step"], saved["epoch"], saved["next_batch"], saved["best_score"]
        restore_rng(saved["rng"])
        del saved
    if step >= training["max_steps"]:
        raise ValueError("Resume checkpoint has already reached max_steps")
    end_step = min(training["max_steps"], step + args.stop_after) if args.stop_after else training["max_steps"]
    model.train()
    log = (run / "events.jsonl").open("x", buffering=1)
    trace = (run / "training-frames.jsonl").open("x", buffering=1)
    started, last_checkpoint, last_saved, checkpoints = time.perf_counter(), None, None, []
    result = dict(status="running", stage=config["stage"])
    def emit(event, **values):
        record = dict(event=event, step=step, seconds=time.perf_counter() - started, **values)
        text = json.dumps(record, allow_nan=False)
        log.write(text + "\n")
        print(text, flush=True)
    def save(is_best=False):
        nonlocal last_checkpoint, last_saved
        if last_saved == step:
            return
        payload = dict(format="adtt-two-stage-v1", config=config, contract=resume_contract(config),
                       manifests=manifests, step=step, epoch=epoch, next_batch=next_batch,
                       best_score=best_score, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), rng=rng_state(), provenance=provenance)
        path = run / f"step-{step:08d}.pt"
        digest = save_checkpoint(path, payload)
        checkpoints.append(path)
        last_checkpoint, last_saved = path, step
        if is_best:
            # best.pt is owned by this fresh run, not a pre-existing user artifact.
            temporary = run / ".best.tmp"
            shutil.copyfile(path, temporary)
            temporary.replace(run / "best.pt")
        while len(checkpoints) > training["keep_checkpoints"]:
            obsolete = checkpoints.pop(0)
            if obsolete.is_symlink() or obsolete.parent != run:
                raise RuntimeError("Unsafe checkpoint retention path")
            obsolete.unlink()
        atomic_json(run / "latest.json", {"checkpoint": path.name, "sha256": digest, "step": step})
        emit("checkpoint", path=str(path), sha256=digest, best=is_best)
    try:
        emit("started", parameters=sum(p.numel() for p in model.parameters()),
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), end_step=end_step,
             torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
             device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
             visible_devices=torch.cuda.device_count() if device.type == "cuda" else 0)
        metrics = evaluate(model, validation_data, config, device, guard)
        atomic_json(run / f"validation-{step:08d}.json", metrics)
        emit("initial_validation", metrics=metrics["aggregate"])
        while step < end_step:
            batches = ResumableBatches(len(training_data), training["batch_size"], training["seed"], epoch, next_batch)
            generator = torch.Generator().manual_seed(training["seed"] + epoch)
            loader = DataLoader(training_data, batch_sampler=batches, collate_fn=collate,
                                num_workers=training["num_workers"], generator=generator)
            cursor = next_batch
            for batch in loader:
                guard.step()
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, components, sampled_counts = compute_training_loss(model, batch, config, device, guard)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["clip_norm"], error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step, cursor = step + 1, cursor + 1
                next_batch = cursor
                if cursor == math.ceil(len(training_data) / training["batch_size"]):
                    epoch, next_batch = epoch + 1, 0
                values = {key: float(value.detach()) for key, value in components.items()}
                trace.write(json.dumps(dict(step=step, indices=batch["indices"],
                    tokens=[[r["token"] for r in rows] for rows in batch["rows"]],
                    frame_content_sha256=batch["frame_content_sha256"],
                    sampled_class_counts=sampled_counts, **values), allow_nan=False) + "\n")
                emit("training", epoch=epoch, next_batch=next_batch, lr=used_lr,
                     next_lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                     step_seconds=time.perf_counter() - tick, **values)
                is_best = False
                if step % training["validate_every"] == 0 or step == end_step:
                    metrics = evaluate(model, validation_data, config, device, guard)
                    atomic_json(run / f"validation-{step:08d}.json", metrics)
                    score = metrics["aggregate"]["semantic_mIoU_nonfree"]
                    if score is not None and (best_score is None or score > best_score):
                        best_score, is_best = score, True
                    emit("validation", metrics=metrics["aggregate"])
                if step % training["checkpoint_every"] == 0 or is_best or step == end_step:
                    save(is_best)
                if step >= end_step:
                    break
        verified = load_checkpoint(last_checkpoint)
        if verified["step"] != step:
            raise RuntimeError("Checkpoint read-back failed")
        result.update(status="budget_complete" if step == training["max_steps"] else "bounded_stop",
                      checkpoint=str(last_checkpoint), best_score=best_score, checkpoint_readback="PASS",
                      validation=metrics, note="Completing the configured budget does not establish convergence")
        emit("finished", status=result["status"], checkpoint=str(last_checkpoint))
    except Exception:
        result.update(status="failed", error=traceback.format_exc(),
                      last_complete_checkpoint=str(last_checkpoint) if last_checkpoint else None)
        raise
    finally:
        result.update(step=step, epoch=epoch, next_batch=next_batch, seconds=time.perf_counter() - started)
        if device.type == "cuda":
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        atomic_json(run / "summary.json", result)
        log.close()
        trace.close()
