#!/usr/bin/env python3
"""Run one STDP prediction from recorded observations; model code is separate."""

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile


VIEWS = ("front", "down", "left", "right")
CLIP_PREFIXES = (
    "stl_encoder.text_embedder.text_model.",
    "image_encoder.backbone.vision_model.",
)


class InputError(ValueError):
    """An actionable error that does not expose local filesystem paths."""


def relative_file(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise InputError("File references must be nonempty relative POSIX paths.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise InputError("Input file references must stay inside their data directory.")
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise InputError("An input file is missing or outside its data directory.")
    return result


def read_observation(args):
    if args.input:
        document = json.loads(Path(args.input).read_text(encoding="utf-8"))
        root = Path(args.input).parent
        stl = document["stl"]
        history = document["history"]
        images = {v: relative_file(root, document["images"][v]) for v in VIEWS}
    else:
        if args.sample_index < 0 or args.step_index < 0:
            raise InputError("Sample and step indices must be nonnegative.")
        record = None
        with Path(args.manifest).open(encoding="utf-8") as stream:
            index = 0
            for line in stream:
                if not line.strip():
                    continue
                if index == args.sample_index:
                    record = json.loads(line)
                    break
                index += 1
        if record is None:
            raise InputError("Sample index exceeds the manifest length.")
        root = Path(args.dataset_root)
        trajectory = relative_file(root, record["trajectory_path"])
        with trajectory.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if args.step_index >= len(rows):
            raise InputError("Step index exceeds the recorded trajectory length.")
        # No future coordinates or images enter the prediction.
        history = [[float(row[k]) for k in ("x", "y", "z")]
                   for row in rows[:args.step_index + 1]]
        current = rows[args.step_index]
        images = {}
        for view in VIEWS:
            filename = current[view + "_image"]
            if not filename or Path(filename).name != filename:
                raise InputError("CSV image entries must contain filenames only.")
            images[view] = relative_file(root, record["images"][view] + "/" + filename)
        stl = record["stl"]
    if not isinstance(stl, str) or not stl.strip():
        raise InputError("A nonempty semantic STL specification is required.")
    if not isinstance(history, list) or not history:
        raise InputError("History must contain at least one observed XYZ position.")
    for point in history:
        if (not isinstance(point, list) or len(point) != 3
                or any(isinstance(x, bool) or not isinstance(x, (float, int))
                       or not math.isfinite(x) for x in point)):
            raise InputError("Every history position must contain three finite numbers.")
    return stl, images, history


def check_images(images):
    from PIL import Image

    for path in images.values():
        with Image.open(path) as image:
            image.convert("RGB").load()


def load_model(checkpoint_path, model_source, clip_model, device):
    source = Path(model_source).resolve()
    if not (source / "src/models/__init__.py").is_file():
        raise InputError(
            "Model implementation is not included in this review release. "
            "Use --model-source with a local implementation; model code is planned "
            "for release upon paper acceptance. --check-inputs works without it."
        )
    import torch

    # The released checkpoint contains tensors and primitive metadata only.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise InputError("This runner requires the released format-version-1 checkpoint.")
    config = copy.deepcopy(checkpoint["config"])
    model_config = config["model"]
    if (model_config.get("type") != "cross_attention_diffusion"
            or model_config.get("point_dim") != 3
            or model_config.get("prediction_horizon") != 5):
        raise InputError("This runner requires the released five-waypoint XYZ model.")
    for section, freeze in (("image", "freeze_backbone"), ("stl", "freeze_clip_text")):
        if not model_config[section].get("pretrained") or not model_config[section].get(freeze):
            raise InputError("The released model requires pretrained, frozen CLIP backbones.")
        model_config[section]["clip_model_name"] = clip_model
    sys.path.insert(0, str(source))
    implementation = importlib.import_module("src.models")
    if not Path(implementation.__file__).resolve().is_relative_to(source):
        raise InputError("A different src.models package is already imported.")
    model = implementation.build_model(config)
    runtime_state = model.state_dict()
    expected = {key for key in runtime_state if not key.startswith(CLIP_PREFIXES)}
    state = checkpoint["model"]
    if set(state) != expected:
        raise InputError("Checkpoint keys do not exactly match the non-CLIP model parameters.")
    # Retain only the separately initialized frozen CLIP parameters; every
    # project-specific parameter must be supplied by the public checkpoint.
    runtime_state.update(state)
    model.load_state_dict(runtime_state, strict=True)
    stats = checkpoint["trajectory_stats"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    if (mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or not (std > 0).all()):
        raise InputError("Checkpoint XYZ normalization statistics are invalid.")
    return model.to(device).eval(), config, mean, std


def prepare_tensors(images, history, image_size, mean, std, device):
    import torch
    from PIL import Image
    from torchvision import transforms

    # Match the published checkpoint's training preprocessing, including view order.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])
    views = []
    for view in VIEWS:
        with Image.open(images[view]) as image:
            views.append(transform(image.convert("RGB")))
    image_tensor = torch.stack(views).unsqueeze(0).to(device)
    history_tensor = (torch.tensor(history, dtype=torch.float32, device=device) - mean) / std
    mask = torch.ones((1, len(history)), dtype=torch.bool, device=device)
    return image_tensor, history_tensor.unsqueeze(0), mask


def predict(args, observation):
    import torch
    import numpy as np

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise InputError("Use cpu, cuda or a CUDA device index.")
    stl, images, history = observation
    model, config, mean, std = load_model(args.checkpoint, args.model_source, args.clip_model, device)
    if len(history) > config["model"]["max_history_len"]:
        raise InputError("History exceeds the checkpoint's supported length; it is not silently truncated.")
    steps = args.steps if args.steps is not None else config["inference"]["ddim_steps"]
    if not 1 <= steps <= config["model"]["diffusion"]["timesteps"]:
        raise InputError("Sampling steps must be within the trained diffusion schedule.")
    inputs = prepare_tensors(images, history, config["data"]["image_size"], mean, std, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        normalized = model.sample([stl], *inputs, steps=steps)
        if normalized.shape != (1, 5, 3) or not torch.isfinite(normalized).all():
            raise InputError("The model did not return five finite XYZ waypoints.")
        waypoints = (normalized[0] * std + mean).cpu().tolist()
    if not all(math.isfinite(x) for point in waypoints for x in point):
        raise InputError("Denormalized waypoints are not finite.")
    return {
        "format_version": 1,
        "coordinate_frame": "released_dataset_xyz",
        "units": "meters",
        "waypoints": waypoints,
        "next_waypoint": waypoints[0],
        "observed_history_length": len(history),
        "sample_steps": steps,
        "seed": args.seed,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
    }


def write_result(destination, result):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".prediction-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Observation JSON; image paths are relative to this file.")
    source.add_argument("--manifest", help="Released split JSONL.")
    parser.add_argument("--dataset-root", default=".", help="Root containing extracted data/.")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based nonempty JSONL record.")
    parser.add_argument("--step-index", type=int, default=0, help="Zero-based recorded CSV row.")
    parser.add_argument("--checkpoint", default="model/STDP.pt")
    parser.add_argument("--model-source", default=".", help="Local root containing src/models/.")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, help="Default: checkpoint setting (20 for this release).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/prediction.json")
    parser.add_argument("--check-inputs", action="store_true", help="Validate observations without model code.")
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.seed < 2**32:
            raise InputError("Seed must be between 0 and 2**32 - 1.")
        observation = read_observation(args)
        check_images(observation[1])
        if args.check_inputs:
            print(json.dumps({"input_check": "passed", "views": list(VIEWS),
                              "observed_history_length": len(observation[2])}))
            return 0
        result = predict(args, observation)
        write_result(args.output, result)
        print("Prediction saved: five XYZ waypoints; no trajectory was executed.")
        return 0
    except InputError as error:
        print("Input error: " + str(error), file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError):
        print("Input/checkpoint could not be read; check files, schema and dependencies.", file=sys.stderr)
    except Exception as error:
        # Avoid publishing local source paths or host/user details in a traceback.
        print("Inference failed (" + type(error).__name__ + "). Check the model implementation, "
              "runtime versions, CLIP files and selected device.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
