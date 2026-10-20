# G-JEPA

### Temporal Context-Aware Physical Grounding in Latent World Models

G-JEPA combines temporal physical grounding with prior-guided action planning. A shared causal temporal module grounds observed and predicted latent sequences in physical states. During planning, a learned action prior supplies proposals that are refined together with Gaussian candidates using the world model.

This codebase is developed from [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm). It includes temporal physical grounding, an action prior, and examples for training, planning, simulator execution, and trajectory visualization.

![G-JEPA overview](assets/overview.png)

[Overview in PDF format](assets/overview.pdf)

## Installation

Use Python 3.11 or 3.12. A CUDA GPU is recommended for training.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Choose the appropriate PyTorch wheel for your CUDA version. For CPU use, install the corresponding CPU wheel. Run the commands below from the repository root.

## Data

Download the image-based HDF5 datasets linked in [LeWorldModel](https://github.com/lucas-maes/le-wm#data) and set their location:

```bash
export GJEPA_DATA_DIR=/path/to/datasets
```

```text
datasets/
├── pusht_expert_train.h5
├── cube_single_expert.h5
└── tworoom.h5
```

The Cube configuration uses the image dataset with physical-state observations. Dataset paths, training budgets, and loss weights are configured under `configs/train/`.

## World-model training

For example, train G-JEPA on PushT:

```bash
python train.py --domain pusht output_dir=runs/pusht/world
```

Use `--domain cube` or `--domain tworooms` with the corresponding dataset. Hydra overrides can change individual settings:

```bash
python train.py --domain cube output_dir=runs/cube/world loader.batch_size=128
```

Training saves model weights, the architecture configuration, normalization statistics, and a checkpoint for resuming. Resume an interrupted run with `resume=true` and the same output directory.

The default configuration enables observed-state grounding, nuisance invariance, and predicted-state grounding. The two temporal branches share parameters and process their sequences separately. The physical-state head is used during training.

### S4D temporal grounding

The temporal module is implemented in [`gjepa/s4d.py`](gjepa/s4d.py): `S4DLayer` performs the causal diagonal recurrence, and `S4DDecoder` stacks two layers with a physical-state readout. The PushT configuration has 55.4K trainable parameters in this head. Observation and prediction sequences use the same decoder with independently initialized recurrent memory. [`gjepa/world_model.py`](gjepa/world_model.py) connects it to the world model, and the training modules apply the two physical-state losses.

## Action-prior training

First cache latent representations using the frozen world model:

```bash
python cache_latents.py --domain pusht \
  --data "$GJEPA_DATA_DIR/pusht_expert_train.h5" \
  --world runs/pusht/world/weights_epoch_20.pt \
  --output cache/pusht

python train_prior.py --cache cache/pusht --output runs/pusht/prior
```

For Cube, use `--domain cube`, `cube_single_expert.h5`, and the corresponding world-model weights. The cache reads the saved training split and action scaler.

The action prior conditions on the current latent, goal latent, and their difference. This implementation uses a diffusion model to generate 25 actions. Training settings such as `--epochs` and `--batch` can be changed through the command line.

## Planning example

Generate a plan from two frames in a dataset episode:

```bash
python plan.py --domain pusht \
  --data "$GJEPA_DATA_DIR/pusht_expert_train.h5" \
  --world runs/pusht/world/weights_epoch_20.pt \
  --prior runs/pusht/prior/best.pt \
  --episode 0 --start 0 --goal-offset 25 \
  --output runs/pusht/plan.json
```

The example mixes learned proposals with Gaussian candidates, scores them with the world model, and saves the selected action sequence in environment action units. Use `--mode cem` without `--prior` for Gaussian CEM. Candidate counts, elite counts, iterations, and the learned-proposal ratio are configurable.

`plan.py` demonstrates one planning round. The simulation example below executes an action prefix and generates a new plan from the updated observation while keeping the goal fixed.

## Simulation and visualization

Install the optional simulator and plotting dependencies:

```bash
pip install -r requirements-sim.txt
```

Run a closed-loop PushT example. Each plan contains 25 actions; this example executes the first five before replanning:

```bash
python simulate.py --domain pusht \
  --data "$GJEPA_DATA_DIR/pusht_expert_train.h5" \
  --world runs/pusht/world/weights_epoch_20.pt \
  --prior runs/pusht/prior/best.pt \
  --episode 0 --start 0 --goal-offset 25 \
  --execute 5 --max-steps 100 --stop-on-success \
  --output runs/pusht/mixed

python plot_rollout.py runs/pusht/mixed \
  --snapshots 4 --gif --output runs/pusht/figures
```

For Cube, use `--domain cube`, `cube_single_expert.h5`, and the Cube world/prior weights. The HDF5 file must include `qpos`, `qvel`, `privileged_block_0_pos`, and `privileged_block_0_quat`. Use `MUJOCO_GL=egl` on a headless Linux machine or `MUJOCO_GL=glfw` on macOS; the simulator selects these defaults automatically.

`--mode cem` runs Gaussian CEM without a prior. `--mode direct` executes a sampled prior sequence without world-model scoring. Run these modes with the same episode, start, and goal, using separate output directories, to compare them:

```bash
python plot_rollout.py runs/pusht/cem runs/pusht/direct runs/pusht/mixed \
  --snapshots 4 --output runs/pusht/comparison
```

Simulation saves `rollout.npz` with executed frames, actions, physical states, and per-step goal errors, plus `summary.json`. Plotting exports trajectory strips and goal-error curves in PNG and editable SVG; `--gif` also exports observation/goal animations. PushT figures draw the recorded physical states with the actual task goal. Use `--rgb` to display original observations, whose green background marker is fixed rather than the task-specific goal.

These are configurable single-task examples. Planning uses terminal latent cost by default; `--cost pusht` enables the PushT staged weighting schedule. `--execute` and `--max-steps` control the execution budget. The examples do not run the paper's full evaluation protocol.

## Loading models

```python
from gjepa.io import load_world, load_prior

world_path = "runs/pusht/world/weights_epoch_20.pt"
world = load_world(world_path, device="cuda")
prior = load_prior("runs/pusht/prior/best.pt", device="cuda", world_path=world_path)
```

Keep `config.json` and `action_scaler.json` beside the world-model weights. The prior includes its latent normalization statistics and checks compatibility with the world model.

Weights are not included in this source repository. Pretrained inference weights can be distributed separately through Hugging Face or GitHub Releases. No public download link is available yet.

## Code structure

```text
gjepa/s4d.py           S4D layers and shared physical-state decoder
gjepa/world_model.py   LeWM-based world model
gjepa/action_prior.py  Conditional action prior
gjepa/planning.py      Mixed-proposal CEM and latent rollout
gjepa/control.py       Image-conditioned planning interface
gjepa/simulation.py    Closed-loop control and simulator adapters
gjepa/envs/            PushT environment
configs/train/         Training configurations
train.py               World-model training
cache_latents.py       Frozen latent extraction
train_prior.py         Action-prior training
plan.py                Single-round planning example
simulate.py            PushT / Cube simulator execution
plot_rollout.py        Trajectory figures, goal errors, and GIF export
assets/                Paper overview
tests/                 Core module and training-entry checks
```

Run `pytest -q` for the CPU checks.

## Acknowledgments

Our world-model implementation is based on [LeWorldModel](https://github.com/lucas-maes/le-wm), with G-JEPA's temporal physical grounding and prior-guided planning extensions. We also use [stable-pretraining](https://github.com/galilai-group/stable-pretraining) and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). The included PushT simulator is adapted from [DINO-WM](https://github.com/gaoyuezhou/dino_wm). Original copyright and license notices are retained in the root [LICENSE](LICENSE).
