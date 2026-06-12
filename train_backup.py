import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from src.data.dataset import PhoenixDataset, collate_fn
from src.models.ctc_model import CSLRModel
from src.utils.metrics import (
word_error_rate,
decode_predictions
)

def validate(model, loader, device, idx2gloss):
    model.eval()


all_preds = []
all_refs = []

with torch.no_grad():

    for frames, input_lengths, flat_labels, label_lengths in loader:

        frames = frames.to(device)

        log_probs = model(frames)

        preds = decode_predictions(
            log_probs,
            idx2gloss
        )

        all_preds.extend(preds)

        offset = 0

        for length in label_lengths:

            ref_tokens = flat_labels[
                offset:offset + length
            ].tolist()

            ref = " ".join(
                idx2gloss.get(t, "<unk>")
                for t in ref_tokens
            )

            all_refs.append(ref)

            offset += length
return word_error_rate(
    all_preds,
    all_refs
)


def train():
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

wandb.init(
    project=cfg["wandb"]["project"],
    config=cfg,
    name=f"resnet50_bilstm_lr{cfg['training']['learning_rate']}"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Training on: {device}")

train_set = PhoenixDataset(
    split="train",
    root_dir=cfg["data"]["dataset_path"],
    max_frames=cfg["data"]["max_frames"]
)

val_set = PhoenixDataset(
    split="val",
    root_dir=cfg["data"]["dataset_path"],
    max_frames=cfg["data"]["max_frames"]
)

train_loader = DataLoader(
    train_set,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_set,
    batch_size=cfg["training"]["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)

model = CSLRModel(
    vocab_size=len(train_set.vocab),
    hidden_size=cfg["model"]["hidden_size"],
    num_layers=cfg["model"]["num_layers"],
    dropout=cfg["model"]["dropout"]
)

model.freeze_backbone(
    freeze=True
)

model = model.to(device)

ctc_loss = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg["training"]["epochs"]
)

os.makedirs(
    "checkpoints",
    exist_ok=True
)

best_wer = float("inf")
start_epoch = 0

latest_ckpt = "checkpoints/latest_model.pt"

if os.path.exists(latest_ckpt):

    print("Resuming from latest checkpoint...")

    ckpt = torch.load(
        latest_ckpt,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    optimizer.load_state_dict(
        ckpt["optimizer_state"]
    )

    best_wer = ckpt["val_wer"]
    start_epoch = ckpt["epoch"] + 1

    print(
        f"Resumed from epoch {start_epoch}"
        f" | WER: {best_wer:.4f}"
    )

for epoch in range(
    start_epoch,
    cfg["training"]["epochs"]
):

    if epoch == cfg["training"]["unfreeze_epoch"]:

        model.freeze_backbone(
            freeze=False
        )

        print(
            "Unfreezing backbone"
        )

    model.train()

    total_loss = 0

    pbar = tqdm(
        train_loader,
        desc=(
            f"Epoch {epoch+1}/"
            f"{cfg['training']['epochs']}"
        )
    )

    for (
        frames,
        input_lengths,
        flat_labels,
        label_lengths
    ) in pbar:

        frames = frames.to(device)
        flat_labels = flat_labels.to(device)

        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        log_probs = model(frames)

        log_probs_t = log_probs.permute(
            1,
            0,
            2
        )

        loss = ctc_loss(
            log_probs_t,
            flat_labels,
            input_lengths,
            label_lengths
        )

        if torch.isnan(loss):
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg["training"]["clip_grad_norm"]
        )

        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = (
        total_loss /
        len(train_loader)
    )

    scheduler.step()

    val_wer = validate(
        model,
        val_loader,
        device,
        train_set.idx2gloss
    )

    wandb.log({
        "train_loss": avg_loss,
        "val_wer": val_wer,
        "lr": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1
    })

    print(
        f"Epoch {epoch+1}"
        f" | Loss: {avg_loss:.4f}"
        f" | Val WER: {val_wer:.4f}"
    )

    if val_wer < best_wer:

        best_wer = val_wer

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_wer": val_wer,
                "vocab": train_set.vocab
            },
            "checkpoints/best_model.pt"
        )

        wandb.save(
            "checkpoints/best_model.pt"
        )

        print(
            f"Saved best model"
            f" | WER: {val_wer:.4f}"
        )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_wer": val_wer,
            "vocab": train_set.vocab
        },
        latest_ckpt
    )

    if (
        (epoch + 1)
        % cfg["training"]["save_every"]
        == 0
    ):
        torch.save(
            model.state_dict(),
            f"checkpoints/epoch_{epoch+1}.pt"
        )

wandb.finish()

print(
    f"Training complete."
    f" Best Val WER: {best_wer:.4f}"
)

if __name__ == "__main__":
    train()
