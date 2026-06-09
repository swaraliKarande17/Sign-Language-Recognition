import os   
import yaml 
import torch    
import torch.nn as nn   
from torch.utils.data import DataLoader 
import wandb    
from tqdm import tqdm   
    
from src.data.dataset import PhoenixDataset, collate_fn 
from src.models.ctc_model import CSLRModel  
from src.utils.metrics import word_error_rate, decode_predictions   
    
    
def validate(model, loader, device, idx2gloss): 
    """Run one full pass over validation set and return WER.""" 
    model.eval()    
    all_preds, all_refs = [], []    
    
    with torch.no_grad():   
        for frames, keypoints, labels, fl, ll in loader:    
            frames = frames.to(device)  
            if keypoints is not None:   
                keypoints = keypoints.to(device)    
    
            log_probs = model(frames, keypoints)    
            preds = decode_predictions(log_probs, idx2gloss)    
            all_preds.extend(preds) 
    
            # Reconstruct reference strings from concatenated label tensor  
            offset = 0  
            for length in ll:   
                ref_tokens = labels[offset:offset + length].tolist()    
                ref = " ".join([idx2gloss.get(t, "<unk>") for t in ref_tokens]) 
                all_refs.append(ref)    
                offset += length    
    
    return word_error_rate(all_preds, all_refs) 
    
    
def train():    
    # ── Load configuration ─────────────────────────────────────────── 
    with open("configs/config.yaml") as f:  
        cfg = yaml.safe_load(f) 
    
    # ── Initialize Weights & Biases ────────────────────────────────── 
    wandb.init( 
        project=cfg["wandb"]["project"],    
        config=cfg, 
        name=f"resnet50_bilstm_kp_lr{cfg['training']['learning_rate']}" 
    )   
    
    # ── Device — use GPU if available ──────────────────────────────── 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    print(f"Training on: {device}") 
    
    # ── Datasets ───────────────────────────────────────────────────── 
    train_set = PhoenixDataset( 
        cfg["data"]["dataset_path"], split="train", 
        max_frames=cfg["data"]["max_frames"],   
        img_size=cfg["data"]["img_size"]    
    )   
    val_set = PhoenixDataset(   
        cfg["data"]["dataset_path"], split="val",   
        max_frames=cfg["data"]["max_frames"],   
        img_size=cfg["data"]["img_size"]    
    )   
    from src.data.dataset import collate_fn

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
    
    
    # ── Model ──────────────────────────────────────────────────────── 
    model = CSLRModel(  
        vocab_size=len(train_set.gloss2idx),    
        hidden_size=cfg["model"]["hidden_size"],    
        num_layers=cfg["model"]["num_layers"],  
        dropout=cfg["model"]["dropout"] 
    )   
    # Freeze backbone initially — train only BiLSTM first   
    model.freeze_backbone(freeze=True)  
    model = model.to(device)    
    
    # ── Loss, Optimizer, Scheduler ─────────────────────────────────── 
    # blank=0 matches the <blank> token index in our vocabulary 
    # zero_infinity=True prevents nan loss on bad batches   
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = torch.optim.AdamW(  
        filter(lambda p: p.requires_grad, model.parameters()),  
        lr=cfg["training"]["learning_rate"],    
        weight_decay=cfg["training"]["weight_decay"]    
    )   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
        optimizer, T_max=cfg["training"]["epochs"]  
    )   
    
    os.makedirs("checkpoints", exist_ok=True)
    best_wer    = float("inf")
    start_epoch = 0

    # Resume from latest checkpoint if it exists
    if os.path.exists("checkpoints/latest_model.pt"):
        print("Resuming from latest checkpoint...")
        ckpt = torch.load("checkpoints/latest_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_wer    = ckpt["val_wer"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch} | WER: {best_wer:.4f}")
    
    # ── Training Loop ──────────────────────────────────────────────── 
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        # Unfreeze CNN backbone after N epochs to fine-tune everything  
        if epoch == cfg["training"]["unfreeze_epoch"]:  
            model.freeze_backbone(freeze=False) 
            print("Unfreezing CNN backbone for fine-tuning")    
    
        model.train()   
        total_loss = 0  
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")    
    
        for frames, keypoints, labels, fl, ll in pbar:  
            frames = frames.to(device)  
            if keypoints is not None:   
                keypoints = keypoints.to(device)    
            labels = labels.to(device)  
    
            # Forward pass  
            log_probs = model(frames, keypoints)  # (B, T, V)   
    
            # CTCLoss expects (T, B, V) format  
            frames, input_lens, flat_labels, label_lens = batch
            log_probs = model(frames)                     # (B, T, V)
            log_probs_t = log_probs.permute(1, 0, 2)     # (T, B, V) for CTC
            loss = ctc_loss(log_probs_t, flat_labels,
                input_lens, label_lens)
            # Skip NaN batches
            if torch.isnan(loss):
                continue


            # Backward pass 
            optimizer.zero_grad()   
            loss.backward() 
            # Clip gradients to prevent them from exploding 
            torch.nn.utils.clip_grad_norm_( 
                model.parameters(), cfg["training"]["clip_grad_norm"]   
            )   
            optimizer.step()    
    
            total_loss += loss.item()   
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})    
    
        avg_loss = total_loss / len(train_loader)   
        scheduler.step()    
    
        # Validate every epoch  
        val_wer = validate(model, val_loader, device, train_set.idx2gloss)  
    
        # Log everything to Weights & Biases dashboard  
        wandb.log({ 
            "train_loss": avg_loss, 
            "val_wer": val_wer, 
            "lr": optimizer.param_groups[0]["lr"],  
            "epoch": epoch + 1  
        })  
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val WER: {val_wer:.4f}")   
    
        # Save checkpoint if this is the best model so far  
        if val_wer < best_wer:  
            best_wer = val_wer  
            torch.save({    
                "epoch": epoch, 
                "model_state": model.state_dict(),  
                "optimizer_state": optimizer.state_dict(),  
                "val_wer": val_wer, 
                "vocab": train_set.gloss2idx    
            }, "checkpoints/best_model.pt") 
            wandb.save("checkpoints/best_model.pt") 
            print(f"  Saved best model — WER: {val_wer:.4f}")   

        
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_wer": val_wer,
            "vocab": train_set.gloss2idx
        }, "checkpoints/latest_model.pt")
    
        # Regular checkpoint every N epochs 
        if (epoch + 1) % cfg["training"]["save_every"] == 0:    
            torch.save(model.state_dict(),  
                       f"checkpoints/epoch_{epoch + 1}.pt") 
    
    wandb.finish()  
    print(f"Training complete. Best Val WER: {best_wer:.4f}")   
    
    
if __name__ == "__main__":  
    train() 