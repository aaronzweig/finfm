from pytorch_lightning.callbacks import Callback
import numpy as np

class DatasetUpdateCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dataset = trainer.train_dataloader.dataset
        if hasattr(dataset, "update"):
            dataset.update(trainer.current_epoch, pl_module.embed_net)

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
