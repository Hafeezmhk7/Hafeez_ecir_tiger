"""
Utility functions for the RecSys project.
"""

# imports
import os
import argparse
import gin
import torch
import random
import numpy as np
from data.schemas import TokenizedSeqBatch
from einops import rearrange
from torch import Tensor
import logging
from datetime import timedelta
from rich import print as rprint
from rich.table import Table
from rich.console import Console
import torch
from torchinfo import summary


# fetch logger
logger = logging.getLogger("recsys_logger")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(type):
    if type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif type == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device =  torch.device("cpu")
    logger.info(f"Using device: {device}")

    return device


def reset_kv_cache(fn):
    def inner(self, *args, **kwargs):
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out

    return inner


def reset_encoder_cache(fn):
    def inner(self, *args, **kwargs):
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        out = fn(self, *args, **kwargs)
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        return out

    return inner


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def select_columns_per_row(x: Tensor, indices: Tensor) -> torch.Tensor:
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]

    B = x.shape[0]
    return x[rearrange(torch.arange(B, device=x.device), "B -> B 1"), indices]


def maybe_repeat_interleave(x, repeats, dim):
    if not isinstance(x, Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)


@torch.no_grad
def compute_debug_metrics(
    batch: TokenizedSeqBatch, model_output=None, prefix: str = ""
) -> dict:
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix
        + f"seq_length_p{q}": torch.quantile(seq_lengths, q=q).detach().cpu().item()
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item()
            for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics


def display_args(args, title="Arguments"):
    """
    Nicely print argparse.Namespace or dict using rich.

    :param:
        args: argparse.Namespace or dict
    :param:
        title: Optional table title
    """
    if not isinstance(args, dict):
        args = vars(args)

    # supress big output
    try:
        args.dataset.label_dict = "[SUPPRESSED]"
    except:
        pass

    table = Table(title=title)
    table.add_column("Argument", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in args.items():
        table.add_row(str(key), str(value))

    console = Console()
    console.print(table)

def display_metrics(metrics: dict, elasped=None, title="Validation Metrics"):
    """
    Nicely print metric dictionary using rich.

    :param:
        metrics: Dictionary of metrics (e.g., {"AUC": 0.95, "ACC": 0.88, ...})
    :param:
        title: Optional table title
    """
    if elasped:
        rprint("Time Elasped:", str(timedelta(seconds=elasped)))
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(str(key), f"{value:.6f}" if isinstance(value, (int, float)) else str(value))

    console = Console()
    console.print(table)


def display_model_summary(model, input_shape=None, title="Model Summary", device="cpu"):
    """
    Display model parameter summary using rich. Supports multiple inputs.

    :param:
        model: PyTorch model
    :param:
        input_shape: A single shape tuple or list/tuple e.g., (1, 3, 224, 224)
    :param:
        title: Optional table title
    :param:
        device: Device for dummy inputs
    """
    model = model.to(device)
    table = Table(title=title)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    table.add_row("Model", model.__class__.__name__)
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Trainable (Million)", f"{trainable_params / (1024 ** 2):.2f} Million")
    arch = str(summary(model, depth=1, verbose=0))
    table.add_row("Summary", arch)


    console = Console()
    console.print(table)
