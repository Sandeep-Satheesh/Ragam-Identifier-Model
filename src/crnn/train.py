import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

from .dataset import CarnaticPitchDataset
from .model import RagamCRNN
from src import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class RagamTrainer:
    def __init__(self, dataset: CarnaticPitchDataset, batch_size, epochs, lr, device, pooling,
                 checkpoint_dir=constants.CHECKPOINT_DIR_PATH,
                 resume=constants.RESUME_TRAINING,
                 early_stopping_patience=constants.EARLY_STOPPING_PATIENCE):
        self.base_dir = dataset.base_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Dataset
        self.dataset = dataset
        val_size = int(0.2 * len(self.dataset))
        train_size = len(self.dataset) - val_size
        self.train_set, self.val_set = random_split(self.dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=6, pin_memory=True, prefetch_factor=8, persistent_workers=True,collate_fn=ragam_collate_fn
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=6, pin_memory=True, prefetch_factor=8, persistent_workers=True,collate_fn=ragam_collate_fn
        )

        # Model
        self.num_ragas = len(self.dataset.raga2idx)
        self.model = RagamCRNN(num_classes=self.num_ragas, pooling=pooling).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Early stopping
        self.best_val_loss = float("inf")
        self.no_improve_epochs = 0

        # TensorBoard
        self.writer = SummaryWriter(log_dir=constants.TENSORBOARD_LOG_DIR)

        # Resume support
        self.start_epoch = 0
        if resume:
            self._load_checkpoint()

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        path = os.path.join(self.checkpoint_dir, f"ragam_epoch{epoch+1}.pth")
        torch.save(state, path)
        logger.info(f"\t--> Saved checkpoint: {path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_path)
            logger.info("\t--> Updated best model.")

    def _load_checkpoint(self):
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("ragam_epoch")]
        if not files:
            logger.info("\t--> No checkpoint found, starting fresh.")
            return
        latest = sorted(files)[-1]
        path = os.path.join(self.checkpoint_dir, latest)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.start_epoch = checkpoint["epoch"] + 1
        logger.info(f"\t--> Resumed from checkpoint {path} at epoch {self.start_epoch}")

    def train(self):
        epoch_pbar = tqdm(range(self.start_epoch, self.epochs), position=0, initial=self.start_epoch, total=self.epochs)
        for epoch in epoch_pbar:
            # --- Train ---
            self.model.train()
            train_loss, correct, total = 0, 0, 0
            epoch_pbar.set_description(f"Epoch [{epoch+1}/{self.epochs}]")

            step_pbar = tqdm(enumerate(self.train_loader, start=1),
                             total=len(self.train_loader), position=1, leave=False)

            for step, (xb, yb) in step_pbar:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, pred = out.max(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

                step_pbar.set_description(
                    f"Step [{step}/{len(self.train_loader)}] "
                    f"Loss={loss.item():.4f}"
                )
                self.writer.add_scalar("Loss/step", loss.item(),
                                       epoch * len(self.train_loader) + step)

            train_acc = correct / total
            avg_train_loss = train_loss / len(self.train_loader)

            # --- Validate ---
            val_loss, val_correct, val_total = 0, 0, 0
            all_preds, all_labels = [], []
            self.model.eval()
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    loss = self.criterion(out, yb)

                    val_loss += loss.item()
                    _, pred = out.max(1)
                    val_correct += (pred == yb).sum().item()
                    val_total += yb.size(0)

                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())

            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(self.val_loader)

            # Logging
            print(f"\nEpoch {epoch+1}: "
                  f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            print('-' * 80)

            self.writer.add_scalar("Loss/train", avg_train_loss, epoch+1)
            self.writer.add_scalar("Loss/val", avg_val_loss, epoch+1)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch+1)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch+1)

            # LR scheduler step
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LR", current_lr, epoch + 1)

            # Early stopping + checkpointing
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.no_improve_epochs = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.no_improve_epochs += 1
                logger.info(f"\t--> No improvement for {self.no_improve_epochs} epochs.")

            if self.no_improve_epochs >= self.early_stopping_patience:
                logger.info(f"\t--> Early stopping at epoch {epoch+1}")
                break

            # Regular checkpoint every N epochs
            if (epoch+1) % constants.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(epoch)

        self.writer.close()

    def log_confusion_matrix(self, labels, preds, epoch):
        cm = confusion_matrix(labels, preds)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        num_ragas = len(self.dataset.raga2idx)
        if num_ragas <= 20:
            classes = list(self.dataset.raga2idx.keys())
            ax.set_xticks(np.arange(num_ragas))
            ax.set_yticks(np.arange(num_ragas))
            ax.set_xticklabels(classes, rotation=90)
            ax.set_yticklabels(classes)

        self.writer.add_figure("ConfusionMatrix", fig, epoch)
        plt.close(fig)

#Picklable function
def ragam_collate_fn(batch):
    """
    Custom collate function for Ragam dataset.
    batch: list of (x, y) from Dataset.__getitem__

    Ensures tensors are stacked correctly and converted to contiguous memory.
    """
    xs, ys = zip(*batch)

    # Stack inputs into (B, 1, T)
    xs = torch.stack(xs, dim=0).contiguous()

    # Stack labels into (B,)
    ys = torch.tensor(ys, dtype=torch.long)

    return xs, ys
