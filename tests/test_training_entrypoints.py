"""Exercise the three public training commands on synthetic HDF5 data."""

import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("domain", ["pusht", "cube", "tworooms"])
def test_training_command_saves_loadable_weights(tmp_path, domain):
    from gjepa.io import load_world

    rng = np.random.default_rng(19)
    names = {
        "pusht": "pusht_expert_train.h5",
        "cube": "cube_single_expert.h5",
        "tworooms": "tworoom.h5",
    }
    with h5py.File(tmp_path / names[domain], "w") as f:
        f["ep_len"] = np.full(10, 30, np.int64)
        f["ep_offset"] = np.arange(10, dtype=np.int64) * 30
        f["pixels"] = rng.integers(0, 256, (300, 32, 32, 3), dtype=np.uint8)
        width = 5 if domain == "cube" else 2
        f["action"] = rng.normal(size=(300, width)).astype(np.float32) * 0.1
        if domain == "cube":
            f["observation"] = rng.normal(size=(300, 28)).astype(np.float32)
        else:
            f["proprio"] = rng.normal(size=(300, 2)).astype(np.float32)
            if domain == "pusht":
                f["state"] = rng.normal(size=(300, 7)).astype(np.float32)
    output = tmp_path / "run"
    command = [
        sys.executable,
        "train.py",
        "--domain",
        domain,
        f"output_dir={output}",
        "trainer.accelerator=cpu",
        "trainer.precision=32-true",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=2",
        "loader.batch_size=2",
        "num_workers=0",
        "stop_after_epochs=1",
    ]
    if domain == "tworooms":
        command.append("trainer.max_epochs=1")
    env = {
        **os.environ,
        "GJEPA_DATA_DIR": str(tmp_path),
        "OMP_NUM_THREADS": "2",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "WANDB_MODE": "disabled",
    }
    result = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert (output / "checkpoints/model_latest.ckpt").is_file()
    model = load_world(output / "weights_epoch_1.pt")
    assert model.state_head is not None
    assert not model.training
    if domain != "pusht":
        split = json.loads((output / "split_manifest.json").read_text())
        assert not set(split["train_episodes"]) & set(split["val_episodes"])
