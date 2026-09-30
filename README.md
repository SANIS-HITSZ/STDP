# STDP: Signal Temporal Logic-Guided Diffusion Policy for UAV Visual Navigation without Global Semantic Maps

> This repository is anonymized for double-blind review.

## Open-Source Progress

[![Dataset](https://img.shields.io/badge/dataset-released-blue)](https://anonymous-hf.com/a/h8or94tutz4u/)
![Model Weights](https://img.shields.io/badge/model_weights-released-blue)
![Code](https://img.shields.io/badge/code-partially_released-yellow)

| Component | Status |
|---|---|
| Dataset | [Released](https://anonymous-hf.com/a/h8or94tutz4u/) |
| Model weights | Released |
| Code | [Inference runner released](#trajectory-inference); model implementation and training code upon paper acceptance |

## Method

We introduce the Signal Temporal Logic-Guided Diffusion Policy (STDP), which
generates receding-horizon three-dimensional (3-D) waypoints from STL
specifications, local multi-view RGB observations, and motion history. It
combines a relation-aware graph Transformer with cross-attention-based
conditional diffusion.

<p align="center">
  <img src="assets/model_overview.png" width="95%" alt="Overview of the STDP framework" />
</p>

### Closed-Loop Planning Algorithm

**Algorithm 1: UAV Visual Navigation without a Global Semantic Map**

```text
Require: STL specification phi, trained policy pi_theta, horizon H,
         diffusion steps T_d, and rollout length K_phi
1:  Parse phi once into typed graph G_phi
2:  Initialize waypoint history h_0 from the episode origin
3:  for k = 0, 1, ..., K_phi - 1 do
4:      Acquire front, down, left, and right RGB views o_k
5:      Encode G_phi, o_k, and h_k as memory M_k
6:      Sample x_Td ~ N(0, I)
7:      for each selected reverse diffusion time t do
8:          epsilon_hat <- Dec_theta(x_t, t, M_k)
9:          Update x_t-1 with the diffusion scheduler
10:     end for
11:     De-standardize and reshape x_0 into H waypoints
12:     Execute the first waypoint; append the reached pose to form h_k+1
13: end for
```

## Experimental Results

STDP achieves 63.6% ID STL success with a 2.7% collision rate and 41.7% OOD
success with a 4.1% collision rate under changed layouts and furniture
placement. Figure 3 shows successful plans for all four STL task families.

<p align="center">
  <img src="assets/main_results.png" width="78%" alt="Closed-loop navigation results" />
</p>

<p align="center">
  <img src="assets/qualitative_results.png" width="95%" alt="Qualitative trajectory-planning results" />
</p>

### Ablation Studies

#### 1. STL Encoder

The graph Transformer increases overall SR<sub>STL</sub> from 38.2% to 63.6% in
ID and from 30.8% to 41.7% in OOD, supporting more effective STL encoding than
CLIP-Text.

<p align="center">
  <img src="assets/ablation_stl_encoder.png" width="76%" alt="STL encoder ablation results" />
</p>

#### 2. Trajectory Model

Cross-attention-based conditional diffusion outperformed FiLM-based
conditional diffusion across scenes and task families, with overall gains of
25.9 points in ID and 10.0 points in OOD.

<p align="center">
  <img src="assets/ablation_trajectory_model.png" width="86%" alt="Trajectory model ablation results" />
</p>

#### 3. DDIM Denoising Steps

Table V shows the performance of different DDIM denoising steps, where 20 steps
outperformed both 10 and 50 steps; we therefore used 20 steps throughout.

<p align="center">
  <img src="assets/ablation_ddim_steps.png" width="86%" alt="DDIM denoising-step ablation results" />
</p>

## Dataset and Model Weights

### Dataset

**Released.** The complete dataset is available on [Anonymous Hugging Face](https://anonymous-hf.com/a/h8or94tutz4u/).
The dataset split manifests are included in this repository. The dataset contains
synchronized Gazebo observations and reference trajectories from four
furnished indoor scenes. Each trajectory includes:

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
trajectory path, and four image-directory paths. The dataset is distributed as
one archive per trajectory. Extracting an archive restores its `data/` paths
relative to the dataset root. To use the dataset, select a split, read its
JSONL records, download and extract the corresponding trajectory archives,
then load the referenced CSV and synchronized image files.

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

## Trajectory Inference

[inference/infer_trajectory.py](inference/infer_trajectory.py) reads a semantic STL
specification, four RGB views and observed trajectory history, then writes five
XYZ waypoints. It does not launch Gazebo, control a vehicle or modify input data.

**The inference runner is released. Model implementation and training code will
be released upon paper acceptance.** Full inference currently requires a
separately supplied local implementation exposing `src.models.build_model(config)`;
the public checkout alone cannot run the model. The `--check-inputs` mode works
without model code.

### Dependencies and weights

Use Python 3.11. From the repository root:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 transformers==4.57.3 \
  numpy==2.4.0 Pillow==12.1.0
python inference/infer_trajectory.py --help
```

Choose a PyTorch build compatible with your CPU or CUDA installation. The
checkpoint excludes the frozen CLIP weights. Supply the standard
`openai/clip-vit-base-patch32` model through the Transformers cache, or put its
files in a local directory and pass `--clip-model ./clip-vit-base-patch32`.
Both the text and vision backbone must use that same pretrained CLIP model.
To prohibit network downloads when the local CLIP files are complete, set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for the invocation.

Model implementation files must be placed locally before full inference, for
example under `./model-source/src/models/`. No model implementation is copied
or generated by this runner. It uses `torch.load(..., weights_only=True)` for
the released checkpoint and requires every project-specific parameter; only
the separately initialized frozen CLIP parameters are absent from that file.

### Use a released trajectory observation

Download and extract the desired trajectory archive from the
[anonymous dataset](https://anonymous-hf.com/a/h8or94tutz4u/). For the first entry
in `splits/test_same.jsonl`, extract
`archives/world_v1/trajectory_0587_05.tar` under `./STDP-Dataset/` so that
`./STDP-Dataset/data/world_v1/...` exists. Download instructions are in the
anonymous dataset's README.

Validate that the observation is readable, without model code:

```bash
python inference/infer_trajectory.py \
  --manifest splits/test_same.jsonl \
  --dataset-root ./STDP-Dataset \
  --sample-index 0 --step-index 2 \
  --check-inputs
```

With the model implementation supplied locally:

```bash
python inference/infer_trajectory.py \
  --checkpoint model/STDP.pt \
  --model-source ./model-source \
  --clip-model ./clip-vit-base-patch32 \
  --manifest splits/test_same.jsonl \
  --dataset-root ./STDP-Dataset \
  --sample-index 0 --step-index 2 \
  --device cuda --seed 42 \
  --output outputs/prediction.json
```

`--sample-index` counts nonempty JSONL records starting at zero. `--step-index`
is a zero-based CSV row. At step 2, only positions from rows 0, 1 and 2 and the
four images referenced by row 2 enter the prediction. Future reference positions
are not inference inputs. The runner does not extend or pad the recorded CSV.

Image and trajectory references inside a manifest must be relative to
`--dataset-root`. References outside that directory are rejected. Command-line
file arguments are resolved from the current working directory; examples use
relative paths.

### Use a custom observation

Alternatively, use `--input observation.json`. Its schema is:

```json
{
  "stl": "F[0:20](chair)",
  "images": {
    "front": "observations/front.png",
    "down": "observations/down.png",
    "left": "observations/left.png",
    "right": "observations/right.png"
  },
  "history": [[-7.0, 0.6, 1.8], [-6.8, 0.6, 1.8]]
}
```

The image paths are relative to the JSON file's directory. The four RGB views
must correspond to the last observed history position. History contains observed
XYZ positions in meters, in the same coordinate convention as the released
trajectory CSVs. This checkpoint was trained on those CSV coordinates: do not
substitute body-frame displacements or independently subtract an episode origin.
The runner applies the checkpoint mean and standard deviation, then reverses
that normalization for the prediction.

```bash
python inference/infer_trajectory.py \
  --input observation.json \
  --model-source ./model-source \
  --clip-model ./clip-vit-base-patch32 \
  --device cpu --output outputs/prediction.json
```

History must contain at least one finite XYZ position and cannot exceed the
checkpoint limit (256 positions). It is not silently truncated. Images are
ordered `front`, `down`, `left`, `right`, resized to 224 by 224, and normalized
with the same CLIP statistics used during training.

### Output and interpretation

The output JSON contains `waypoints` (five XYZ positions), `next_waypoint`
(the first position), the observed history length, sampling steps, seed and
checkpoint SHA256. It contains no input paths, usernames or hostnames.
The default is the checkpoint's 20 sampling steps; `--steps` may override it
within the trained diffusion schedule. Results may differ across hardware or
software versions even with the same seed.

This is one prediction conditioned on recorded observations. Repeated predictions
against a recorded dataset are not a closed-loop rollout. In a closed-loop system,
execute only the first waypoint, obtain new synchronized observations and actual
executed history, then replan. The runner neither executes waypoints nor performs
collision checking or STL-success evaluation; its output alone is not evidence
of safe flight or reproduction of the paper's benchmark results.

## Citation

Coming soon.
