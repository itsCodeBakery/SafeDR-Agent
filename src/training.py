
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        all_targets.extend(labels.detach().cpu().numpy())
        all_predictions.extend(preds.detach().cpu().numpy())

        current_loss = running_loss / len(all_targets)
        current_acc = accuracy_score(all_targets, all_predictions)

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}"
        })

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_predictions)
    epoch_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, desc="Validation"):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_targets.extend(labels.detach().cpu().numpy())
        all_predictions.extend(preds.detach().cpu().numpy())
        all_probabilities.extend(probs.detach().cpu().numpy())

        current_loss = running_loss / len(all_targets)
        current_acc = accuracy_score(all_targets, all_predictions)

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}"
        })

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_predictions)
    epoch_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    epoch_precision = precision_score(all_targets, all_predictions, average="macro", zero_division=0)
    epoch_recall = recall_score(all_targets, all_predictions, average="macro", zero_division=0)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "macro_f1": epoch_f1,
        "macro_precision": epoch_precision,
        "macro_recall": epoch_recall,
        "targets": np.array(all_targets),
        "predictions": np.array(all_predictions),
        "probabilities": np.array(all_probabilities)
    }
