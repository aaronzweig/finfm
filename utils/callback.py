from pytorch_lightning.callbacks import Callback
import numpy as np

class DatasetUpdateCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dataset = trainer.train_dataloader.dataset
        if hasattr(dataset, "update"):
            dataset.update(trainer.current_epoch, pl_module.embed_net)

class BestModelCallback(Callback):
    """Tracks training loss each epoch and restores the best model weights at the end of training."""
    def __init__(self, monitor="train_loss"):
        self.monitor = monitor
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.best_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get(self.monitor)
        if loss is None:
            return
        loss_val = loss.item()
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_epoch = trainer.current_epoch
            self.best_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}

    def on_train_end(self, trainer, pl_module):
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)
            print(f"Restored best model from epoch {self.best_epoch} with {self.monitor}={self.best_loss:.6f}")

class MeanChangeEarlyStopping(Callback):
    def __init__(self, window_size=10, delta_threshold=1e-4, min_epochs=20):
        self.window_size = window_size
        self.delta_threshold = delta_threshold
        self.min_epochs = min_epochs
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is None:
            return

        self.train_losses.append(loss.item())

        if len(self.train_losses) < 2 * self.window_size or len(self.train_losses) < self.min_epochs:
            return

        prev_window = self.train_losses[-2*self.window_size : -self.window_size]
        curr_window = self.train_losses[-self.window_size:]
        delta = abs(np.mean(curr_window) - np.mean(prev_window))

        if delta < self.delta_threshold:
            print(f"\nEarly stopping triggered. Δmean(train_loss) = {delta:.2e} < {self.delta_threshold}")
            trainer.should_stop = True
