from datetime import datetime
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_ROOT))
sys.path.append(str(CODE_ROOT / "utils"))
sys.path.append(str(CODE_ROOT / "models"))

from config import *
from dataset import ImageGeolocationDataset, collate_fn
from losses import (
    build_categorical_loss_weights,
    haversine_distance_km,
    multitask_loss,
    unit_vector_loss,
)
from model import SUPPORTED_ARCHITECTURES, build_model
from printers import print_config


def make_target(batch, device):
    return torch.stack(
        [
            batch["latitude"].to(device, non_blocking=True),
            batch["longitude"].to(device, non_blocking=True),
        ],
        dim=1,
    )


def make_loss_weights():
    return {
        "coord": COORD_LOSS_WEIGHT,
        "dist_sea": DIST_SEA_LOSS_WEIGHT,
        "country": COUNTRY_LOSS_WEIGHT,
        "climate": CLIMATE_LOSS_WEIGHT,
        "land_cover": LAND_COVER_LOSS_WEIGHT,
        "expert_diversity": EXPERT_DIVERSITY_LOSS_WEIGHT,
        "expert_load_balance": EXPERT_LOAD_BALANCE_LOSS_WEIGHT,
    }


def get_num_classes(dataset):
    return {
        "num_country_classes": len(dataset.label_vocab["country"]),
        "num_climate_classes": len(dataset.label_vocab["climate"]),
        "num_land_cover_classes": len(dataset.label_vocab["land_cover"]),
    }


def get_xyz(preds):
    return preds["xyz"] if isinstance(preds, dict) else preds


def compute_loss(preds, batch, target, categorical_loss_weights, loss_weights):
    if MODEL_ARCHITECTURE == "base":
        loss = unit_vector_loss(preds, target)
        return loss, loss
    return multitask_loss(
        preds, batch, target, categorical_loss_weights, loss_weights
    )


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch,
                    categorical_loss_weights, loss_weights):
    model.train()
    total_loss = running_loss = 0.0
    total_samples = running_samples = 0
    use_amp = device.type == "cuda"

    for step, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        target = make_target(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(images)
            loss, coord_loss = compute_loss(
                preds, batch, target, categorical_loss_weights, loss_weights
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += coord_loss.item() * batch_size
        total_samples += batch_size
        running_loss += coord_loss.item() * batch_size
        running_samples += batch_size

        if step % LOG_EVERY_STEPS == 0 or step == len(dataloader):
            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(dataloader):05d} | "
                f"recent unit-vector loss: {running_loss / running_samples:.4f} | "
                f"epoch avg: {total_loss / total_samples:.4f}",
                flush=True,
            )
            running_loss = 0.0
            running_samples = 0

    return total_loss / total_samples


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = total_km = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        target = make_target(batch, device)
        pred_xyz = get_xyz(model(images))
        loss = unit_vector_loss(pred_xyz, target)
        distances_km = haversine_distance_km(pred_xyz, target)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_km += distances_km.sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_km / total_samples


def build_dataloaders():
    train_dataset = ImageGeolocationDataset(TRAIN_ROOT, LABEL_VOCAB_PATH)
    eval_dataset = ImageGeolocationDataset(EVAL_ROOT, LABEL_VOCAB_PATH)
    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(eval_dataset, shuffle=False, **common),
    )


def main():
    if MODEL_ARCHITECTURE not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"MODEL_ARCHITECTURE must be one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    print_config()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders()
    train_dataset = train_loader.dataset
    is_multitask = MODEL_ARCHITECTURE != "base"
    categorical_loss_weights = (
        build_categorical_loss_weights(train_dataset, CATEGORICAL_LABELS)
        if is_multitask else None
    )
    loss_weights = make_loss_weights() if is_multitask else None

    model = build_model(
        MODEL_ARCHITECTURE,
        base_channels=BASE_CHANNELS,
        dropout=DROPOUT,
        num_experts=NUM_EXPERTS,
        gate_temperature=GATE_TEMPERATURE,
        **get_num_classes(train_dataset),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            categorical_loss_weights, loss_weights,
        )
        val_loss, val_km = validate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | train unit-vector loss: {train_loss:.4f} | "
            f"val unit-vector loss: {val_loss:.4f} | val avg error: {val_km:.2f} km | "
            f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_architecture": MODEL_ARCHITECTURE,
            "model_config": {
                "base_channels": BASE_CHANNELS,
                "dropout": DROPOUT,
                "num_experts": NUM_EXPERTS,
                "gate_temperature": GATE_TEMPERATURE,
                **get_num_classes(train_dataset),
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_km": val_km,
            "label_vocab_path": str(LABEL_VOCAB_PATH),
        }
        torch.save(checkpoint, LAST_MODEL_PATH)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"Saved best model to {BEST_MODEL_PATH}")

    print(f"Best validation unit-vector loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
