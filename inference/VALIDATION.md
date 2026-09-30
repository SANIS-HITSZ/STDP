# Inference runner validation

Validated on September 30, 2026, using the released `model/STDP.pt` and actual
observations extracted from the anonymous dataset. The model implementation was
provided locally and is not part of this review release. Hostnames, usernames,
local directories and raw machine logs are omitted from this public record.

## Runtime

| Component | Validated version |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.9.1 |
| torchvision | 0.24.1 |
| Transformers | 4.57.3 |
| NumPy | 2.4.0 |
| Pillow | 12.1.0 |
| Inference devices | CUDA (GeForce RTX 3080) and CPU |

The public checkpoint SHA256 was:
`9668d7837246fc4d76316159e97bf9352676f4a92953ebb01574f9569f6e8a92`.

Pretrained CLIP files were loaded locally with network downloads disabled.
Sampling used 20 steps and seed 42. Each sample used recorded row 2, its four
RGB images, and three observed history positions (rows 0 through 2).

## Actual model runs

| Scene | Released trajectory | Result |
|---|---|---|
| world_v1 | trajectory_0587_05 | Five finite XYZ waypoints on CUDA |
| world_v2 | trajectory_0131_05 | Five finite XYZ waypoints on CUDA |
| world_v3 | trajectory_0131_05 | Five finite XYZ waypoints on CUDA |
| world_v4 | trajectory_0131_05 | Five finite XYZ waypoints on CUDA |
| world_v1 | trajectory_0587_05, explicit observation JSON | Five finite XYZ waypoints on CPU and CUDA |

For the world_v1 observation, the manifest and explicit JSON input modes produced
identical CUDA output. The new runner's image tensors, normalized history and
history mask matched the existing research preprocessing exactly. With the same
seed and loaded model, the existing inference interface and new runner produced
identical denormalized waypoints (maximum absolute difference: 0.0).

Additional checks confirmed:

- Input validation works without the model implementation.
- Full inference without model code exits with an explanation of the release
  status and does not create a prediction file.
- Absolute image references and parent-directory traversal are rejected.
- Removing a project-specific parameter from the checkpoint causes loading to
  fail; omitted frozen CLIP weights are handled separately.
- The 41 existing research Python source files retained their original hashes.

## Scope

These are real checkpoint inference checks, not placeholder or synthetic-model
outputs. They validate the new entry point with the local model implementation.
They do not make the withheld model implementation available to reviewers.
Only the listed observations were tested; no full benchmark rerun, closed-loop
Gazebo run or physical flight was performed for this release. CPU and CUDA
outputs are not claimed to be bitwise identical.
