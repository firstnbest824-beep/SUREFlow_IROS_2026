"""Minimal OpenVLA checkpoint verification for LIBERO-Spatial.

Loads the official OpenVLA LIBERO-Spatial fine-tuned checkpoint and verifies
that required dataset statistics and action tokenizer metadata are present.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify OpenVLA checkpoint for LIBERO-Spatial")
    parser.add_argument(
        "--checkpoint_id",
        type=str,
        default="openvla/openvla-7b-finetuned-libero-spatial",
        help="Hugging Face checkpoint identifier",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="962318cec55ac10993ff0f5f43eda9a270b4c873",
        help="Checkpoint revision/commit hash",
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="libero_spatial",
        help="Dataset statistics key to verify",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention implementation passed to from_pretrained",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/hwkim/env-audit/openvla-vanilla-rollout/checkpoint_info.json",
        help="Where to write checkpoint verification JSON",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    start = time.time()
    print(f"Loading processor from {args.checkpoint_id} @ {args.revision}")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint_id,
        revision=args.revision,
        trust_remote_code=True,
    )
    proc_load_time = time.time() - start
    print(f"Processor loaded in {proc_load_time:.2f}s")

    start = time.time()
    print(f"Loading model from {args.checkpoint_id} @ {args.revision}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint_id,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model_load_time = time.time() - start
    print(f"Model loaded in {model_load_time:.2f}s")
    model.eval()

    print("Model class:", type(model).__name__)
    print("Model dtype:", next(model.parameters()).dtype)

    # Verify action tokenizer / norm stats.
    action_tokenizer = getattr(model, "action_tokenizer", None)
    if action_tokenizer is None and hasattr(model, "module"):
        action_tokenizer = getattr(model.module, "action_tokenizer", None)

    norm_stats: Optional[Dict[str, Any]] = None
    if hasattr(model, "norm_stats"):
        norm_stats = model.norm_stats
    elif hasattr(model, "module") and hasattr(model.module, "norm_stats"):
        norm_stats = model.module.norm_stats

    info: Dict[str, Any] = {
        "checkpoint_id": args.checkpoint_id,
        "revision": args.revision,
        "model_class": type(model).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "attn_implementation": args.attn_implementation,
        "processor_class": type(processor).__name__,
        "processor_load_time_sec": proc_load_time,
        "model_load_time_sec": model_load_time,
        "unnorm_key_requested": args.unnorm_key,
    }

    if action_tokenizer is not None:
        info["action_tokenizer"] = {
            "class": type(action_tokenizer).__name__,
            "num_bins": getattr(action_tokenizer, "num_bins", None),
            "action_dim": getattr(action_tokenizer, "action_dim", None),
            "vocab_size": getattr(action_tokenizer, "vocab_size", None),
        }
        print("Action tokenizer:", info["action_tokenizer"])
    else:
        print("WARNING: action_tokenizer not found on model")

    if norm_stats is not None:
        info["norm_stats_keys"] = list(norm_stats.keys())
        info["unnorm_key_present"] = args.unnorm_key in norm_stats
        print("Norm stats keys:", info["norm_stats_keys"])
        print(f"Requested unnorm_key '{args.unnorm_key}' present:", info["unnorm_key_present"])
        if info["unnorm_key_present"]:
            stats = norm_stats[args.unnorm_key]
            info["unnorm_key_stats"] = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in stats.items()
            }
            print("Stats:", info["unnorm_key_stats"])
    else:
        print("WARNING: norm_stats not found on model")
        info["norm_stats_keys"] = []
        info["unnorm_key_present"] = False

    # Minimal inference smoke test with dummy input to confirm action shape.
    try:
        from transformers.image_utils import load_image
        from PIL import Image
        import numpy as np

        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        prompt = "In: What action should the robot take to pick up the soup and place it in the basket?\nOut:"
        inputs = processor(prompt, dummy_image).to("cuda", dtype=dtype)

        with torch.inference_mode():
            action_start = time.time()
            generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            action_time = time.time() - action_start

        predicted_action = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        info["dummy_inference_time_sec"] = action_time
        info["dummy_decoded_output"] = predicted_action
        print(f"Dummy inference OK in {action_time:.2f}s")
    except Exception as exc:  # noqa: BLE001
        print("Dummy inference failed:", exc)
        info["dummy_inference_error"] = str(exc)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Wrote checkpoint info to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
