"""TabM scalar-horizon training and inference adapted from tabm4pv.py."""

from __future__ import annotations

import json
import math
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import sklearn.preprocessing
import tabm
import torch
import torch.nn.functional as F

from .features import build_horizon_features, build_horizon_target, feature_names


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "history_length": 96,
    "label_scale": 500.0,
    "prediction_clip_lower": 0.0,
    "prediction_clip_upper": 465.0,
    "learning_rate": 0.002,
    "weight_decay": 0.0003,
    "gradient_clipping_norm": 1.0,
    "epochs": 200,
    "patience": 10,
    "batch_size": 512,
    "inference_batch_size": 512,
    "noise_std": 0.00001,
    "n_blocks": 2,
    "d_block": 512,
    "dropout": 0.1,
    "k": 32,
    "embedding_dim": 32,
    "device": "auto",
}


def resolved_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {**DEFAULT_CONFIG, **(config or {})}
    if result["label_scale"] <= 0:
        raise ValueError("label_scale must be positive")
    if result["epochs"] <= 0 or result["patience"] < 0:
        raise ValueError("epochs must be positive and patience must be non-negative")
    return result


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config["device"])
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _make_model(n_features: int, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(
        n_features,
        d_embedding=int(config["embedding_dim"]),
    )
    return tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        num_embeddings=embeddings,
        n_blocks=int(config["n_blocks"]),
        d_block=int(config["d_block"]),
        dropout=float(config["dropout"]),
        k=int(config["k"]),
    ).to(device)


def _predict_tensor(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    label_scale: float,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(values, device=device, dtype=torch.float32)
    outputs = []
    with torch.inference_mode():
        for batch in tensor.split(batch_size):
            ensemble = model(batch, None).squeeze(-1).float()
            outputs.append(ensemble.mean(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32) * float(label_scale)


def _fit_preprocessor(x_train: np.ndarray, config: dict[str, Any], seed: int):
    noise = np.random.default_rng(seed).normal(
        0.0,
        float(config["noise_std"]),
        x_train.shape,
    ).astype(x_train.dtype)
    n_quantiles = max(min(x_train.shape[0] // 30, 1000), 10)
    return sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=10**9,
        random_state=seed,
    ).fit(x_train + noise)


def _train_horizon(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    horizon_index: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, float | int]]:
    seed = int(config["seed"]) + horizon_index
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    x_train = build_horizon_features(
        train_frame,
        horizon_index,
        history_length=int(config["history_length"]),
    )
    x_valid = build_horizon_features(
        validation_frame,
        horizon_index,
        history_length=int(config["history_length"]),
    )
    y_train = build_horizon_target(train_frame, horizon_index)
    y_valid = build_horizon_target(validation_frame, horizon_index)
    preprocessor = _fit_preprocessor(x_train, config, seed)
    x_train = preprocessor.transform(x_train).astype(np.float32)
    x_valid = preprocessor.transform(x_valid).astype(np.float32)
    train_x = torch.as_tensor(x_train, device=device)
    train_y = torch.as_tensor(y_train / float(config["label_scale"]), device=device)

    model = _make_model(x_train.shape[1], config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_state = deepcopy(model.state_dict())
    best_validation_mse = math.inf
    best_epoch = -1
    remaining_patience = int(config["patience"])
    batch_size = int(config["batch_size"])
    for epoch in range(int(config["epochs"])):
        model.train()
        batches = torch.randperm(len(train_x), device=device).split(batch_size)
        for indices in batches:
            optimizer.zero_grad()
            ensemble_prediction = model(train_x[indices], None).squeeze(-1).float()
            repeated_target = train_y[indices].repeat_interleave(model.backbone.k)
            loss = F.mse_loss(ensemble_prediction.flatten(), repeated_target)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clipping_norm"]),
            )
            optimizer.step()
        validation_prediction = _predict_tensor(
            model,
            x_valid,
            device=device,
            batch_size=int(config["inference_batch_size"]),
            label_scale=float(config["label_scale"]),
        )
        validation_mse = float(np.mean(np.square(validation_prediction - y_valid)))
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            remaining_patience = int(config["patience"])
        else:
            remaining_patience -= 1
            if remaining_patience < 0:
                break
    model.load_state_dict(best_state)
    return model, preprocessor, {
        "best_epoch": best_epoch,
        "validation_mse": best_validation_mse,
    }


def train(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    checkpoint_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one scalar TabM per benchmark horizon without reading data files."""

    cfg = resolved_config(config)
    checkpoint_root = Path(checkpoint_dir)
    models_dir = checkpoint_root / "models"
    preprocessors_dir = checkpoint_root / "preprocessors"
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessors_dir.mkdir(parents=True, exist_ok=True)
    horizon_points = len(np.asarray(train_frame.iloc[0]["target"]))
    device = _device(cfg)
    horizon_metrics = []
    for horizon_index in range(horizon_points):
        model, preprocessor, training_info = _train_horizon(
            train_frame,
            validation_frame,
            horizon_index,
            cfg,
            device,
        )
        torch.save(model.state_dict(), models_dir / f"horizon_{horizon_index + 1:02d}.pt")
        with (preprocessors_dir / f"horizon_{horizon_index + 1:02d}.pkl").open("wb") as stream:
            pickle.dump(preprocessor, stream)
        horizon_metrics.append({"horizon": horizon_index + 1, **training_info})
        print(
            f"horizon={horizon_index + 1:02d}/{horizon_points} "
            f"best_epoch={training_info['best_epoch']} "
            f"validation_mse={training_info['validation_mse']:.6f}"
        )
    metadata = {
        "model": "tabm4pv_16_scalar_models",
        "feature_names": feature_names(int(cfg["history_length"])),
        "horizon_points": horizon_points,
        "config": cfg,
        "training_by_horizon": horizon_metrics,
    }
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def predict(frame: pd.DataFrame, checkpoint_dir: str | Path) -> np.ndarray:
    """Predict all benchmark horizons without reading or writing data files."""

    checkpoint_root = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_root / "metadata.json").read_text(encoding="utf-8"))
    cfg = resolved_config(metadata["config"])
    device = _device(cfg)
    predictions = []
    for horizon_index in range(int(metadata["horizon_points"])):
        features = build_horizon_features(
            frame,
            horizon_index,
            history_length=int(cfg["history_length"]),
        )
        with (checkpoint_root / "preprocessors" / f"horizon_{horizon_index + 1:02d}.pkl").open("rb") as stream:
            preprocessor = pickle.load(stream)
        features = preprocessor.transform(features).astype(np.float32)
        model = _make_model(features.shape[1], cfg, device)
        state = torch.load(
            checkpoint_root / "models" / f"horizon_{horizon_index + 1:02d}.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        predictions.append(
            _predict_tensor(
                model,
                features,
                device=device,
                batch_size=int(cfg["inference_batch_size"]),
                label_scale=float(cfg["label_scale"]),
            )
        )
    result = np.column_stack(predictions)
    return np.clip(
        result,
        float(cfg["prediction_clip_lower"]),
        float(cfg["prediction_clip_upper"]),
    ).astype(np.float32)
