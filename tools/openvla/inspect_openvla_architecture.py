#!/usr/bin/env python3
"""Inspect OpenVLA model architecture and capture intermediate representations.

Loads the OpenVLA checkpoint, prints its module tree, and runs a single dummy
forward pass through `predict_action` while capturing shapes/dtypes of key
intermediate tensors with forward hooks.

Python 3.8 compatible syntax only.
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", "/home/hwkim/.config/vla-spatial-diagnostics/libero"
)

DEFAULT_CHECKPOINT = "openvla/openvla-7b-finetuned-libero-spatial"
DEFAULT_REVISION = "962318cec55ac10993ff0f5f43eda9a270b4c873"
DEFAULT_UNNORM_KEY = "libero_spatial"
DEFAULT_GPU = 0
DEFAULT_OUTPUT_DIR = "/home/hwkim/env-audit/openvla-architecture-inspection"


def log_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def get_top_level_attributes(obj: Any) -> Dict[str, str]:
    """Return public attribute names and their types/classes."""
    result: Dict[str, str] = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        if isinstance(attr, torch.nn.Module):
            result[name] = type(attr).__module__ + "." + type(attr).__name__
        elif isinstance(attr, (str, int, float, bool, type(None))):
            result[name] = f"{type(attr).__name__}={attr!r}"
    return result


def summarize_named_modules(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Collect named modules and their parameter counts."""
    summary: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        summary.append({
            "module_name": name,
            "class": type(module).__module__ + "." + type(module).__name__,
            "params": params,
        })
    return summary


def select_hook_candidates(summary: List[Dict[str, Any]]) -> List[str]:
    """Select module names that likely correspond to functional stages."""
    interesting_classes = {
        "CLIPVisionModel", "CLIPVisionTransformer",
        "Dinov2Model", "Dinov2Encoder", "Dinov2PatchEmbeddings",
        "SiglipVisionModel", "SiglipVisionTransformer",
        "LlamaModel", "LlamaDecoderLayer",
        "PrismaticVLM", "PrismaticVisionBackbone",
        "VisionBackbone", "Projector",
        "Linear",
    }
    candidates: List[str] = []
    for info in summary:
        cls_name = info["class"].split(".")[-1]
        if cls_name in interesting_classes:
            candidates.append(info["module_name"])
    # Also include a few high-level names if they exist.
    for name in ["vision_backbone", "vision_tower", "vision_model", "vision_projector",
                 "projector", "language_model", "lm_head", "action_head", "model"]:
        if any(s["module_name"] == name for s in summary):
            candidates.append(name)
    # De-duplicate while preserving order.
    seen = set()
    result: List[str] = []
    for name in candidates:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


class ActivationRecorder:
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self._handles: List[Any] = []

    def register(self, model: torch.nn.Module, module_names: List[str]) -> None:
        name_to_module: Dict[str, torch.nn.Module] = {}
        for n, m in model.named_modules():
            name_to_module[n] = m
        for name in module_names:
            module = name_to_module.get(name)
            if module is None:
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
            try:
                tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(tensor, torch.Tensor):
                    return
                record = {
                    "module_name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "has_nan": bool(torch.isnan(tensor).any().item()),
                    "has_inf": bool(torch.isinf(tensor).any().item()),
                }
                # If tensor has spatial/patch structure, record that.
                if len(tensor.shape) >= 3:
                    record["is_spatial"] = True
                else:
                    record["is_spatial"] = False
                self.records[name] = record
            except Exception as exc:
                self.records[name] = {"error": str(exc)}
        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def load_openvla(
    checkpoint_id: str,
    revision: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Any, Any]:
    print(f"[*] Loading processor from {checkpoint_id} @ {revision}")
    processor = AutoProcessor.from_pretrained(
        checkpoint_id,
        revision=revision,
        trust_remote_code=True,
    )
    print(f"[*] Loading model from {checkpoint_id} @ {revision}")
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    return processor, vla


def build_dummy_inputs(
    processor: Any,
    task_label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    inputs = processor(prompt, image)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
    if input_ids[0, -1].item() != 29871:
        empty_token = torch.tensor([[29871]], dtype=input_ids.dtype, device=device)
        attend_token = torch.tensor([[1]], dtype=attention_mask.dtype, device=device)
        input_ids = torch.cat([input_ids, empty_token], dim=1)
        attention_mask = torch.cat([attention_mask, attend_token], dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect OpenVLA architecture")
    parser.add_argument("--checkpoint_id", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    parser.add_argument("--unnorm_key", type=str, default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--task_label", type=str, default="pick up the black bowl and place it on the plate")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "inspection.log")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout

    class Tee:
        def write(self, msg: str) -> None:
            original_stdout.write(msg)
            log_file.write(msg)
        def flush(self) -> None:
            original_stdout.flush()
            log_file.flush()

    sys.stdout = Tee()
    sys.stderr = Tee()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    metadata: Dict[str, Any] = {
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "device": str(device),
        "dtype": args.dtype,
        "gpu": args.gpu,
    }

    try:
        log_section("Environment")
        print(f"Python: {sys.executable}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

        log_section("Loading OpenVLA")
        processor, vla = load_openvla(
            checkpoint_id=args.checkpoint_id,
            revision=args.revision,
            dtype=dtype,
            device=device,
        )
        metadata["model_class"] = type(vla).__module__ + "." + type(vla).__name__
        metadata["processor_class"] = type(processor).__module__ + "." + type(processor).__name__
        print(f"Model class: {metadata['model_class']}")
        print(f"Processor class: {metadata['processor_class']}")

        log_section("Top-level model attributes")
        top_attrs = get_top_level_attributes(vla)
        for k, v in top_attrs.items():
            print(f"  {k}: {v}")
        metadata["top_level_attributes"] = top_attrs

        log_section("Action dimension")
        action_dim = vla.get_action_dim(args.unnorm_key)
        print(f"Action dimension: {action_dim}")
        metadata["action_dim"] = action_dim

        log_section("Module summary")
        module_summary = summarize_named_modules(vla)
        print(f"Total modules: {len(module_summary)}")
        # Print modules with non-zero local parameters or interesting classes.
        for info in module_summary:
            cls_short = info["class"].split(".")[-1]
            if info["params"] > 0 or cls_short in {
                "CLIPVisionModel", "Dinov2Model", "SiglipVisionModel",
                "LlamaModel", "LlamaDecoderLayer", "Linear",
            }:
                print(f"  {info['module_name']}: {info['class']} (params={info['params']:,})")
        metadata["module_summary"] = module_summary

        hook_candidates = select_hook_candidates(module_summary)
        log_section(f"Hook candidates ({len(hook_candidates)})")
        for name in hook_candidates:
            print(f"  {name}")
        metadata["hook_candidates"] = hook_candidates

        log_section("Running dummy forward pass with hooks")
        inputs = build_dummy_inputs(processor, args.task_label, device, dtype)
        print(f"input_ids shape: {list(inputs['input_ids'].shape)}")
        print(f"attention_mask shape: {list(inputs['attention_mask'].shape)}")
        print(f"pixel_values shape: {list(inputs['pixel_values'].shape)}")

        recorder = ActivationRecorder()
        recorder.register(vla, hook_candidates)

        torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            action = vla.predict_action(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                unnorm_key=args.unnorm_key,
                do_sample=False,
            )
        elapsed = time.time() - start
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else None

        recorder.remove()

        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        print(f"predict_action output shape: {list(action_np.shape)}, dtype: {action_np.dtype}")
        print(f"predict_action output: {action_np}")
        print(f"Inference latency: {elapsed * 1000:.2f} ms")
        if peak_mem_mb is not None:
            print(f"Peak VRAM: {peak_mem_mb:.2f} MB")

        metadata["predict_action_output"] = {
            "shape": list(action_np.shape),
            "dtype": str(action_np.dtype),
            "sample": action_np.flatten().tolist(),
        }
        metadata["latency_ms"] = elapsed * 1000
        metadata["peak_vram_mb"] = peak_mem_mb

        log_section("Hook records")
        for name, record in recorder.records.items():
            print(f"  {name}: {record}")
        metadata["hook_records"] = recorder.records

        # Also try a raw forward() call to see if it exposes different internals.
        log_section("Raw forward() outputs")
        try:
            with torch.no_grad():
                forward_out = vla.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                )
            if isinstance(forward_out, dict):
                print("forward() returned dict keys:", sorted(forward_out.keys()))
                for k, v in forward_out.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={list(v.shape)}, dtype={str(v.dtype)}")
                metadata["forward_output_keys"] = sorted(forward_out.keys())
            else:
                print(f"forward() returned type: {type(forward_out)}")
                metadata["forward_output_type"] = type(forward_out).__name__
        except Exception as exc:
            print(f"forward() call failed: {exc}")
            metadata["forward_error"] = str(exc)

        metadata["success"] = True

    except Exception as exc:
        metadata["success"] = False
        metadata["exception"] = traceback.format_exc()
        print("\n[ERROR] Exception during inspection:")
        traceback.print_exc()

    finally:
        metadata_path = os.path.join(args.output_dir, "architecture_inspection.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\nMetadata saved: {metadata_path}")
        log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stdout

    return 0 if metadata.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
