# STDP: Signal Temporal Logic-Guided Diffusion Policy for UAV Visual Navigation without Global Semantic Maps

> This repository is anonymized for double-blind review.

## Open-Source Progress

![Dataset](https://img.shields.io/badge/dataset-released-brightgreen)
![Model Weights](https://img.shields.io/badge/model_weights-released-blue)
![Code](https://img.shields.io/badge/code-coming_soon-lightgrey)

| Component | Status |
|---|---|
| Dataset | Released |
| Model weights | Released |
| Source code | Coming soon |

## Method

STDP generates receding-horizon 3D waypoints directly from an STL
specification, synchronized local multi-view RGB observations, and executed
motion history, without access to a task-level global semantic map.

The method parses each STL specification into a typed syntax graph. A
relation-aware graph Transformer preserves operator types, temporal intervals,
predicate semantics, syntactic hierarchy, and directed operand relations.
Frozen CLIP encoders extract visual and object-name features, while a trajectory
encoder represents the executed waypoint history. A multimodal Transformer
fuses these condition tokens, and a cross-attention diffusion decoder predicts
the next five 3D waypoints. During closed-loop execution, only the first
waypoint is executed before the observations and history are updated for the
next prediction.

<p align="center">
  <img src="assets/model_overview.png" width="95%" alt="Overview of the STDP framework" />
</p>

## Experimental Results

The following table reports closed-loop STL success rate and collision rate in
the in-distribution scene and three out-of-distribution scenes.

<p align="center">
  <img src="assets/main_results.png" width="78%" alt="Closed-loop navigation results" />
</p>

The examples below show successful closed-loop plans for Reach-Avoid,
Multi-Target, Either-Or, and Door Puzzle tasks. Solid blue and dashed gray
curves denote executed and reference trajectories; green and orange overlays
denote target and avoidance regions.

<p align="center">
  <img src="assets/qualitative_results.png" width="95%" alt="Qualitative trajectory-planning results" />
</p>

## Released Dataset and Model Weights

### Dataset

The `data/` directory contains synchronized Gazebo observations and reference
trajectories from four furnished indoor scenes. Each trajectory includes:

- a semantic STL specification and its coordinate-grounded form;
- synchronized `front`, `down`, `left`, and `right` RGB observations; and
- a CSV trajectory containing absolute XYZ positions, camera yaw, frame
  indices, and image filenames.

The `splits/` directory provides the following manifests:

- `train.jsonl`: training data from the primary scene;
- `validation.jsonl`: validation data from the primary scene;
- `test_same.jsonl`: evaluation data from the primary scene; and
- `test_cross.jsonl`: cross-layout evaluation data from the remaining scenes.

Each JSONL record contains the STL formula, task type, scene identifier,
trajectory path, and four image-directory paths. All paths are relative to the
repository root. To use the dataset, select a split, read its JSONL records,
then load the referenced trajectory CSV and synchronized image files.

### Model Weights

`model/STDP.pt` contains the Stage 1 weights used for evaluation. The checkpoint
includes:

- the project-specific model state;
- the model and inference configuration;
- trajectory normalization statistics; and
- checkpoint metadata.

The frozen CLIP vision and text parameters are intentionally excluded. The
required `openai/clip-vit-base-patch32` model must be downloaded separately or
provided through a local `transformers` cache. The checkpoint also excludes
optimizer state and local filesystem paths.

Loading the weights requires the forthcoming source code. The expected loading
sequence is:

1. Instantiate the STDP architecture from the stored configuration.
2. Initialize the frozen CLIP vision and text encoders from
   `openai/clip-vit-base-patch32`.
3. Load the project-specific state from `checkpoint["model"]`.
4. Use `checkpoint["trajectory_stats"]` to normalize motion history and
   denormalize predicted waypoints.

For inference, provide a semantic STL formula, four synchronized RGB views,
and the executed XYZ history. The model predicts a five-waypoint window in
normalized coordinates. Convert it to absolute XYZ coordinates with the stored
mean and standard deviation, execute the first waypoint, update the inputs, and
repeat.

## Citation

Coming soon.
