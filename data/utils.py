from data.schemas import SeqBatch
import logging
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from collections import Counter
import torch

# fetch logger
logger = logging.getLogger("recsys_logger")


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    return SeqBatch(*[v.to(device) for _, v in batch._asdict().items()])


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)


def describe_dataloader(dataloader, batch_sampler=None, title="DataLoader Summary"):
    """
    Method to print dataset statistics from a PyTorch DataLoader:
        - Total number of samples
        - Total number of batches
        - Class distribution (if available)
        - Sample data shape and dtype
    """
    console = Console()
    dataset = dataloader.dataset
    total_samples = len(dataset)
    total_batches = len(dataloader)

    table = Table(title=title)

    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total samples", str(total_samples))
    if batch_sampler is None:
        batch_size = dataloader.batch_size
    else:
        batch_size = getattr(batch_sampler, "batch_size", "Unknown")
    table.add_row(f"Total batches (batch_size={batch_size})", str(total_batches))
    table.add_row(f"Num Workers", str(dataloader.num_workers))

    # class info
    class_info_found = False
    if hasattr(dataset, 'classes'):
        table.add_row("Classes", str(dataset.classes))
        class_info_found = True
    if hasattr(dataset, 'class_to_idx'):
        table.add_row("Class to index mapping", str(dataset.class_to_idx))
        class_info_found = True
    if hasattr(dataset, 'data_dict') and 'label' in dataset.data_dict:
        targets = dataset.data_dict['label']
        if isinstance(targets, tuple):
            targets = list(targets)
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        label_counts = Counter(targets.tolist())
        table.add_row("Label counts", str(dict(label_counts)))
        class_info_found = True
    if not class_info_found:
        table.add_row("Class/Label info", "No class/label info found in dataset attributes.")

    # sample data shape and dtype
    try:
        first_batch = next(iter(dataloader))
        if isinstance(first_batch, (list, tuple)):
            # Show shape of first input and sample label summary
            shape_info = str(first_batch[0].shape)
            batch_len = str(len(first_batch))
            table.add_row("Input sample shape", shape_info)
            table.add_row("Batch Len", batch_len)
        elif isinstance(first_batch, dict):
            table.add_row("Sample keys", str(list(first_batch.keys())))
            for key, value in first_batch.items():
                if hasattr(value, 'shape'):
                    shape = tuple(value.shape)
                else:
                    shape = 'N/A'
                dtype = getattr(value, 'dtype', type(value).__name__)
                table.add_row(f"{key.capitalize()} shape & dtype", str(shape) + f", ({str(dtype)})")
        else:
            table.add_row("Sample", str(type(first_batch)))
    except Exception as e:
        table.add_row("Sample inspection error", str(e))

    console.print(table)


# =============================================================================
# MULTIMODAL BATCH UTILITIES - CROSS-ATTENTION EXTENSION
# =============================================================================

def multimodal_collate_fn(batch_list):
    """
    Custom collate function that handles both SeqBatch and MultimodalSeqBatch.
    Use this when feature_combination_mode="individual_signals"
    """
    if len(batch_list) == 0:
        return None
        
    first_item = batch_list[0]
    
    # Import here to avoid circular imports
    from data.schemas import MultimodalSeqBatch
    
    if isinstance(first_item, MultimodalSeqBatch):
        # Stack multimodal batches
        try:
            return MultimodalSeqBatch(
                user_ids=torch.stack([b.user_ids for b in batch_list]),
                ids=torch.stack([b.ids for b in batch_list]),
                ids_fut=torch.stack([b.ids_fut for b in batch_list]),
                x_text=torch.stack([b.x_text for b in batch_list]),
                x_image=torch.stack([b.x_image for b in batch_list]),
                x_fut_text=torch.stack([b.x_fut_text for b in batch_list]),
                x_fut_image=torch.stack([b.x_fut_image for b in batch_list]),
                x_brand_id=torch.stack([b.x_brand_id for b in batch_list]),
                x_fut_brand_id=torch.stack([b.x_fut_brand_id for b in batch_list]),
                seq_mask=torch.stack([b.seq_mask for b in batch_list]),
            )
        except Exception as e:
            logger.error(f"Error collating multimodal batch: {e}")
            logger.error(f"First item types: {[(k, type(v), getattr(v, 'shape', 'no shape')) for k, v in first_item._asdict().items()]}")
            raise
    else:
        # Use default collate for SeqBatch
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch_list)

def multimodal_batch_to(batch, device):
    """
    Move batch to device, handling both SeqBatch and MultimodalSeqBatch
    """
    from data.schemas import MultimodalSeqBatch
    
    if isinstance(batch, MultimodalSeqBatch):
        return MultimodalSeqBatch(*[v.to(device) if hasattr(v, 'to') else v for v in batch])
    else:
        return SeqBatch(*[v.to(device) if hasattr(v, 'to') else v for v in batch])

def next_multimodal_batch(dataloader, device):
    """
    Updated batch getter that handles multimodal batches
    """
    batch = next(dataloader)
    return multimodal_batch_to(batch, device)