from torch_lr_finder import LRFinder

from . import DEVICE
from .trainer import Trainer


def find_lr(trainer: Trainer, plot: bool = True) -> float:
    lr_finder = LRFinder(
        trainer.model,
        trainer.optimizer,
        trainer.criterion,
        device=DEVICE,
    )

    lr_finder.range_test(
        trainer.train_loader,
        start_lr=1e-6,
        end_lr=1e-3,
        num_iter=200,
        step_mode="exp",  # exponential increase
        smooth_f=0.05,  # light smoothing
        diverge_th=4,  # early stop if loss > 4x best
    )

    # Suggested LR:
    suggested_max_lr = (
        lr_finder.history["lr"][
            int(lr_finder.history["loss"].index(min(lr_finder.history["loss"])))
        ]
        # * 0.5
    )
    if plot:
        lr_finder.plot()

    return suggested_max_lr