#!/usr/bin/env python3
"""
Test script for Phase 1: Schema extensions


"""


print("🧪 Testing Phase 1: Schema Extensions...")

import sys
import os

# Add parent directory to path so we can import 'data' module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# # Now the imports should work
# from data.schemas import MultimodalSeqBatch, is_multimodal_batch, get_text_features, get_image_features

def test_schemas():
    """Test MultimodalSeqBatch creation and utilities"""
    print("\n1️⃣ Testing MultimodalSeqBatch import...")
    
    try:
        from data.schemas import MultimodalSeqBatch, is_multimodal_batch, get_text_features, get_image_features
        import torch
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    print("\n2️⃣ Testing MultimodalSeqBatch creation...")
    try:
        # Create dummy multimodal batch
        batch = MultimodalSeqBatch(
            user_ids=torch.zeros(2, dtype=torch.long),
            ids=torch.zeros(2, 1, dtype=torch.long),
            ids_fut=torch.zeros(2, 1, dtype=torch.long),
            x_text=torch.randn(2, 768),  # Text features
            x_image=torch.randn(2, 512), # Image features  
            x_fut_text=torch.randn(2, 768),
            x_fut_image=torch.randn(2, 512),
            x_brand_id=torch.zeros(2, 1, dtype=torch.long),
            x_fut_brand_id=torch.zeros(2, 1, dtype=torch.long),
            seq_mask=torch.ones(2, 1, dtype=torch.bool)
        )
        
        print(f"✅ MultimodalSeqBatch created")
        print(f"   - Text shape: {batch.x_text.shape}")
        print(f"   - Image shape: {batch.x_image.shape}")
        print(f"   - Batch size: {batch.user_ids.shape[0]}")
        
    except Exception as e:
        print(f"❌ MultimodalSeqBatch creation failed: {e}")
        return False
    
    print("\n3️⃣ Testing utility functions...")
    try:
        # Test is_multimodal_batch
        is_multi = is_multimodal_batch(batch)
        assert is_multi == True, "is_multimodal_batch should return True"
        
        # Test get_text_features
        text_feats = get_text_features(batch)
        assert text_feats.shape == batch.x_text.shape, "get_text_features shape mismatch"
        
        # Test get_image_features  
        image_feats = get_image_features(batch)
        assert image_feats.shape == batch.x_image.shape, "get_image_features shape mismatch"
        
        print("✅ Utility functions working")
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return False
    
    return True

def test_collate_functions():
    """Test multimodal collate functions"""
    print("\n4️⃣ Testing multimodal collate functions...")
    
    try:
        from data.utils import multimodal_collate_fn, multimodal_batch_to
        from data.schemas import MultimodalSeqBatch
        import torch
        
        # Create list of multimodal batches (simulating dataloader output)
        batch_list = []
        for i in range(3):  # 3 samples
            batch = MultimodalSeqBatch(
                user_ids=torch.tensor([i]),
                ids=torch.tensor([[i]]),
                ids_fut=torch.tensor([[i+1]]),
                x_text=torch.randn(1, 768),
                x_image=torch.randn(1, 512),
                x_fut_text=torch.randn(1, 768),
                x_fut_image=torch.randn(1, 512),
                x_brand_id=torch.tensor([[i]]),
                x_fut_brand_id=torch.tensor([[i+1]]),
                seq_mask=torch.ones(1, 1, dtype=torch.bool)
            )
            batch_list.append(batch)
        
        # Test collate function
        collated = multimodal_collate_fn(batch_list)
        
        assert collated.x_text.shape == (3, 1, 768), f"Expected (3, 768), got {collated.x_text.shape}"
        assert collated.x_image.shape == (3, 1, 512), f"Expected (3, 512), got {collated.x_image.shape}"
        
        print("✅ multimodal_collate_fn working")
        print(f"   - Collated text shape: {collated.x_text.shape}")
        print(f"   - Collated image shape: {collated.x_image.shape}")
        
        # Test batch_to function
        device = "cpu"  # Use CPU for testing
        moved_batch = multimodal_batch_to(collated, device)
        assert moved_batch.x_text.device.type == device
        
        print("✅ multimodal_batch_to working")
        
    except Exception as e:
        print(f"❌ Collate function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all Phase 1 tests"""
    print("=" * 50)
    print("🚀 PHASE 1 TESTING: SCHEMA EXTENSIONS")
    print("=" * 50)
    
    success = True
    
    # Test schemas
    success &= test_schemas()
    
    # Test collate functions
    success &= test_collate_functions()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 PHASE 1 COMPLETE: All tests passed!")
        print("✅ Ready to proceed to Phase 2 (Data Loading)")
    else:
        print("❌ PHASE 1 FAILED: Fix errors before proceeding")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    main()