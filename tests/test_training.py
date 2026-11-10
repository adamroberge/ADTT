"""CPU-only qualification of training, resume, geometry and export.

These tiny synthetic grids test implementation correctness, not model quality or
published scores. No real dataset, remote server, GPU or shared environment is used.
"""
import ast
import copy
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import yaml

from adtt_training.config import load_config, validate_config
from adtt_training.data import OccupancyDataset, collate, load_frame
from adtt_training.device import DeviceGuard, GPUConflict, select_device
from adtt_training.engine import compute_training_loss, dense_prediction
from adtt_training.inference import straight_plan
from adtt_training.losses import occupancy_loss, lovasz_softmax_flat
from adtt_training.manifest import MetadataReader, prepare_manifest, read_manifest, sha256
from adtt_training.metrics import Confusion
from adtt_training.models import (build_model, controls_from_poses, encode_sequence,
                                  forecast, make_metas, query_logits, transfer_representation)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def tiny_config():
    config = load_config(ROOT / "configs/stage1_static.yaml")
    config["data"]["grid_size"] = [8, 8, 4]
    config["representation"].update(channels=2, decoder_hidden=16, decoder_layers=2)
    config["forecast"].update(history=2, future=2, token_grid=[2, 2, 2], token_dim=8,
                              depth=1, heads=2, dropout=0.0, ego_queries=2, external_queries=6)
    config["sampling"]["points"] = 32
    config["training"].update(max_steps=4, batch_size=2, checkpoint_every=2,
                               validate_every=2, validation_frames=1, decode_chunk=64,
                               keep_checkpoints=2)
    return config


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="adtt-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.config = tiny_config()
        self.manifests = {}
        self.raw = self.directory / "raw"
        for split, scene in (("train", "scene-0001"), ("val", "scene-0003")):
            rows = []
            for i in range(6):
                token = f"{split}-{i}"
                folder = self.raw / scene / token
                folder.mkdir(parents=True)
                labels = np.full((8, 8, 4), 9, dtype=np.uint8)
                labels[:4, :4, :2] = i % 9
                labels.reshape(-1)[:10] = np.arange(10, dtype=np.uint8)
                np.savez_compressed(folder / "labels.npz", semantics=labels)
                rows.append(dict(scene=scene, token=token, prev=f"{split}-{i-1}" if i else "",
                    timestamp=1000000 + i * 500000, semantics_path=str(folder / "labels.npz"),
                    ego2global_translation=[i * 0.1, 0.0, 0.0], ego2global_rotation=[1.0, 0.0, 0.0, 0.0]))
            path = self.directory / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.manifests[split] = path
            self.config["data"][split + "_manifest"] = str(path)


class DataTests(Fixture):
    def test_windows_stay_inside_scenes_and_masks_are_explicit(self):
        config = copy.deepcopy(self.config)
        config["stage"] = "forecast"
        combined = self.directory / "combined.jsonl"
        combined.write_text(self.manifests["train"].read_text() + self.manifests["val"].read_text())
        dataset = OccupancyDataset(combined, config)
        self.assertEqual(len(dataset), 6)
        for item in dataset:
            self.assertEqual(len({r["scene"] for r in item["rows"]}), 1)
            self.assertEqual(item["labels"].shape, (4, 8, 8, 4))
            self.assertTrue(item["mask"].all())
        config["data"]["visibility"] = "camera"
        with self.assertRaisesRegex(ValueError, "mask_camera"):
            OccupancyDataset(self.manifests["train"], config)[0]

    def test_invalid_labels_and_broken_chains_fail(self):
        rows = read_manifest(self.manifests["train"])
        np.savez(rows[0]["semantics_path"], semantics=np.full((8, 8, 4), 255, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Invalid input label"):
            load_frame(rows[0], self.config["data"])
        rows[1]["prev"] = "wrong-token"
        bad = self.directory / "bad.jsonl"
        bad.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Broken temporal chain"):
            read_manifest(bad)

    def test_flat_numpy_metadata_can_be_converted_without_raw_writes(self):
        rows = read_manifest(self.manifests["train"])
        infos = [dict(row, occ_path=f"gts/{row['scene']}/{row['token']}",
                      ego2global_translation=np.asarray(row["ego2global_translation"])) for row in rows]
        source = self.directory / "metadata.pkl"
        source.write_bytes(pickle.dumps({"infos": infos}))
        before = {str(path): sha256(path) for path in self.raw.rglob("*.npz")}
        output = self.directory / "prepared.jsonl"
        receipt = prepare_manifest(source, self.raw, output)
        self.assertEqual(receipt["frames"], 6)
        self.assertEqual(len(read_manifest(output)), 6)
        self.assertEqual(before, {str(path): sha256(path) for path in self.raw.rglob("*.npz")})
        with self.assertRaises(FileExistsError):
            prepare_manifest(source, self.raw, output)


class ModelTests(Fixture):
    def setUp(self):
        super().setUp()
        torch.manual_seed(7)
        self.ae = build_model(self.config)
        self.config["stage"] = "forecast"
        self.predictor = build_model(self.config)
        transfer_representation(self.predictor, self.ae.state_dict())

    def test_transfer_freezing_and_real_forecast_backward(self):
        dataset = OccupancyDataset(self.manifests["train"], self.config)
        batch = collate([dataset[0], dataset[1]])
        history = batch["labels"][:, :2]
        for a, b in zip(encode_sequence(self.ae, history), encode_sequence(self.predictor, history)):
            torch.testing.assert_close(a, b)
        self.predictor.train()
        loss, _, _ = compute_training_loss(self.predictor, batch, self.config, torch.device("cpu"), DeviceGuard())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in (self.predictor.triplane_encoder, self.predictor.triplane_class_embeds):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        for module in (self.predictor.semantic_decoder, self.predictor.triplane_transformer,
                       self.predictor.control_token_mlp, self.predictor.external_control_norm):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any((g != 0).any() for g in gradients))

    def test_history_api_matches_released_forward_and_ignores_future_occupancy(self):
        # Load only the original function's AST, avoiding duplicate model registry imports.
        source = subprocess.check_output(["git", "show", "c62fcd2:model/TransDelta.py"], cwd=ROOT, text=True)
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeltaTokenPredictorV21")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "released-forward", "exec"), namespace)
        legacy = types.MethodType(namespace["forward"], self.predictor)
        self.predictor.eval()
        batch = collate([OccupancyDataset(self.manifests["train"], self.config)[0]])
        metas = make_metas(batch["rows"])
        with torch.no_grad():
            old = legacy(batch["labels"], None, None, None, None, metas=metas)["hexplane"]
            new = forecast(self.predictor, batch["labels"][:, :2], metas, 2)
            perturbed = batch["labels"].clone()
            perturbed[:, 2:] = torch.randint(0, 10, perturbed[:, 2:].shape)
            changed = legacy(perturbed, None, None, None, None, metas=metas)["hexplane"]
        for a, b, c in zip(old, new, changed):
            torch.testing.assert_close(a, b[0])
            torch.testing.assert_close(a, c)
        with self.assertRaisesRegex(ValueError, "exactly the observed"):
            forecast(self.predictor, batch["labels"], metas, 2)

    def test_query_chunks_preserve_logits_and_gradients(self):
        first = self.ae
        second = build_model(dict(self.config, stage="representation"))
        second.load_state_dict(first.state_dict())
        labels = torch.randint(0, 10, (1, 1, 8, 8, 4))
        queries = torch.rand(1, 32, 3) * 2 - 1
        a = query_logits(first, [p[:, 0] for p in encode_sequence(first, labels)], queries, 32)
        b = query_logits(second, [p[:, 0] for p in encode_sequence(second, labels)], queries, 8, True)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(first.parameters(), second.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)

    def test_controls_pose_interval_and_missing_metadata(self):
        rotations = torch.tensor([[1., 0., 0., 0.], [math.sqrt(.5), 0., 0., math.sqrt(.5)],
                                  [math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        translations = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        controls = controls_from_poses(rotations, translations, 1, 2)
        torch.testing.assert_close(controls, torch.tensor([[1., 0., math.pi/2], [1., 0., 0.]]), atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            self.predictor._build_control_features(None, 2, "cpu")
        plane = torch.randn(1, 2, 4, 4)
        identity = torch.eye(3)[None]
        translation = torch.zeros(1, 3)
        warped = self.predictor._warp_plane_between_ego_frames(plane, identity, translation,
                                                              identity, translation, (0, 1))
        torch.testing.assert_close(plane, warped)


class MetricAndGuardTests(unittest.TestCase):
    def test_lovasz_matches_released_values_and_gradients(self):
        import itertools
        from torch.autograd import Variable
        source = (ROOT / "loss/recon_loss.py").read_text()
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                     and node.name in {"lovasz_grad", "lovasz_softmax_flat", "mean", "isnan"}]
        namespace = dict(torch=torch, Variable=Variable, ifilterfalse=itertools.filterfalse)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "released-lovasz", "exec"), namespace)
        first = torch.randn(30, 10, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        labels = torch.arange(30) % 7
        expected = namespace["lovasz_softmax_flat"](first.softmax(-1), labels)
        actual = lovasz_softmax_flat(second.softmax(-1), labels)
        torch.testing.assert_close(expected, actual)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(first.grad, second.grad)

    def test_false_positive_only_classes_score_zero(self):
        metric = Confusion([str(i) for i in range(10)], 9)
        metric.update(torch.tensor([1, 9]), torch.tensor([0, 9]))
        result = metric.result()
        self.assertEqual(result["geometry_IoU"], 1)
        self.assertEqual(result["semantic_mIoU_nonfree"], 0)
        self.assertEqual(result["per_class_IoU"]["1"], 0)
        self.assertIsNone(result["per_class_IoU"]["2"])

    def test_losses_are_finite_and_exclude_ignore_targets(self):
        logits = torch.randn(5, 10, requires_grad=True)
        value, _ = occupancy_loss(logits, torch.tensor([0, 1, 9, 255, 2]), {"ce": 1, "lovasz": 1})
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[3] == 0).all())

    def test_guard_blocks_compute_graphics_and_conflicting_visibility(self):
        uuid = "GPU-test"
        def xml(pid, kind):
            return f'''<nvidia_smi_log><gpu><uuid>{uuid}</uuid>
              <fb_memory_usage><used>500 MiB</used><free>32000 MiB</free></fb_memory_usage>
              <utilization><gpu_util>0 %</gpu_util></utilization><processes>
              <process_info><pid>{pid}</pid><type>{kind}</type><process_name>python</process_name>
              <used_memory>20 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'''
        for kind in ("C", "G", "C+G"):
            with mock.patch("subprocess.check_output", return_value=xml(123, kind)), mock.patch.object(
                    DeviceGuard, "_process", return_value={"pid": 123}):
                with self.assertRaises(GPUConflict):
                    DeviceGuard(uuid).check()
        with mock.patch("subprocess.check_output", return_value=xml(os.getpid(), "C")), mock.patch.object(
                DeviceGuard, "_process", return_value={"pid": os.getpid()}):
            DeviceGuard(uuid).check(active=True)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-another"}):
            with self.assertRaises(GPUConflict):
                select_device("cuda", uuid)


class EndToEndTests(Fixture):
    def run_command(self, *arguments, success=True):
        result = subprocess.run([sys.executable, "-B", *map(str, arguments)], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        if success and result.returncode:
            self.fail(result.stdout[-3000:] + result.stderr[-6000:])
        if not success:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_both_stages_resume_evaluation_and_100_frame_export(self):
        raw_before = {str(path): sha256(path) for path in self.raw.rglob("*.npz")}
        config_file = self.directory / "stage1.yaml"
        config_file.write_text(yaml.safe_dump(self.config))
        full = self.directory / "stage1-full"
        partial = self.directory / "stage1-partial"
        resumed = self.directory / "stage1-resumed"
        self.run_command("train.py", "--config", config_file, "--work-dir", full)
        self.run_command("train.py", "--config", config_file, "--work-dir", partial, "--stop-after", 2)
        self.run_command("train.py", "--config", config_file, "--work-dir", resumed,
                         "--resume", partial / "step-00000002.pt")
        a = torch.load(full / "step-00000004.pt", weights_only=True)
        b = torch.load(resumed / "step-00000004.pt", weights_only=True)
        for name in a["model"]:
            self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
        self.assertEqual((a["epoch"], a["next_batch"]), (b["epoch"], b["next_batch"]))
        changed = copy.deepcopy(self.config)
        changed["training"]["lr"] *= 2
        changed_file = self.directory / "changed.yaml"
        changed_file.write_text(yaml.safe_dump(changed))
        rejected = self.run_command("train.py", "--config", changed_file,
            "--work-dir", self.directory / "invalid-resume", "--resume", partial / "step-00000002.pt", success=False)
        self.assertIn("mathematical recipe", rejected.stderr)
        self.config["stage"] = "forecast"
        self.config["training"].update(max_steps=2, batch_size=1)
        forecast_config = self.directory / "stage2.yaml"
        forecast_config.write_text(yaml.safe_dump(self.config))
        forecast_run = self.directory / "stage2"
        self.run_command("train.py", "--config", forecast_config, "--work-dir", forecast_run,
                         "--representation-checkpoint", full / "step-00000004.pt")
        checkpoint = forecast_run / "step-00000002.pt"
        state = torch.load(checkpoint, weights_only=True)
        for key, value in a["model"].items():
            for prefix, destination in (("encoder.", "triplane_encoder."), ("class_embeds.", "triplane_class_embeds.")):
                if key.startswith(prefix):
                    self.assertTrue(torch.equal(value, state["model"][destination + key[len(prefix):]]))
        partial_forecast, resumed_forecast = self.directory / "stage2-partial", self.directory / "stage2-resumed"
        self.run_command("train.py", "--config", forecast_config, "--work-dir", partial_forecast,
                         "--representation-checkpoint", full / "step-00000004.pt", "--stop-after", 1)
        self.run_command("train.py", "--config", forecast_config, "--work-dir", resumed_forecast,
                         "--resume", partial_forecast / "step-00000001.pt")
        replay = torch.load(resumed_forecast / "step-00000002.pt", weights_only=True)
        for name in state["model"]:
            torch.testing.assert_close(state["model"][name], replay["model"][name], atol=1e-7, rtol=1e-6)
        evaluation = self.directory / "evaluation"
        self.run_command("evaluate.py", "--checkpoint", checkpoint, "--val-manifest", self.manifests["val"],
                         "--work-dir", evaluation)
        metrics = json.loads((evaluation / "metrics.json").read_text())
        self.assertEqual(metrics["aggregate"]["voxels"], 3 * 2 * 8 * 8 * 4)
        export = self.directory / "export"
        self.run_command("rollout.py", "--checkpoint", checkpoint, "--manifest", self.manifests["val"],
                         "--scene", "scene-0003", "--work-dir", export, "--trajectory", "straight",
                         "--speed-mps", 1, "--frames", 100, "--embedding-policy", "repeat-training-horizon")
        predicted = np.load(export / "occupancy/scene-0003/pred.npy", allow_pickle=False)
        ground_truth = np.load(export / "occupancy/scene-0003/gt.npy", allow_pickle=False)
        self.assertEqual(predicted.shape, (100, 8, 8, 4))
        self.assertEqual(ground_truth.shape[0], 4)
        self.assertTrue(set(np.unique(predicted)).issubset({0, 1, 8, 11, 12, 13, 14, 15, 16, 17}))
        metadata = json.loads((export / "metadata.json").read_text())
        self.assertFalse(metadata["paired_with_GT"])
        self.assertEqual(raw_before, {str(path): sha256(path) for path in self.raw.rglob("*.npz")})
        self.run_command("train.py", "--config", config_file, "--work-dir", full, success=False)


if __name__ == "__main__":
    unittest.main()
