from typing import NamedTuple
from torch import Tensor

FUT_SUFFIX = "_fut"


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut_brand_id: Tensor
    x_fut: Tensor
    x_brand_id: Tensor
    seq_mask: Tensor


class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor

# =============================================================================
# MULTIMODAL BATCH SUPPORT - CROSS-ATTENTION EXTENSION
# =============================================================================

class MultimodalSeqBatch(NamedTuple):
    """Extended batch format for individual text and image features"""
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x_text: Tensor           # Separate text features [B, text_dim] or [B, S, text_dim]
    x_image: Tensor          # Separate image features [B, image_dim] or [B, S, image_dim]
    x_fut_text: Tensor       # Future text features
    x_fut_image: Tensor      # Future image features
    x_brand_id: Tensor
    x_fut_brand_id: Tensor
    seq_mask: Tensor

# Utility functions for batch handling
def is_multimodal_batch(batch) -> bool:
    """Check if batch contains separate text/image features"""
    return isinstance(batch, MultimodalSeqBatch)

def get_batch_size(batch) -> int:
    """Get batch size from either batch type"""
    return batch.user_ids.shape[0]

def get_text_features(batch):
    """Extract text features regardless of batch type"""
    if is_multimodal_batch(batch):
        return batch.x_text
    else:
        return batch.x  # Traditional combined features

def get_image_features(batch):
    """Extract image features from multimodal batch"""
    if is_multimodal_batch(batch):
        return batch.x_image
    else:
        return None  # No separate image features
