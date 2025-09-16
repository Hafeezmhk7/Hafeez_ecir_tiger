#!/usr/bin/env python3
"""
Working Phase 2 Test - Building on successful minimal test
"""

print("Starting Phase 2 test...")

import sys
import os

# Add parent directory to path so we can import 'data' module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import torch
    import numpy as np
    from data.schemas import SeqBatch, MultimodalSeqBatch, is_multimodal_batch
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print("\n🧪 PHASE 2 TEST: Individual Signals Mode")
print("=" * 50)

def test_individual_signals_concept():
    """Test the core concept of individual_signals mode"""
    print("\n1️⃣ Testing individual_signals concept...")
    
    try:
        # Simulate what ItemData with individual_signals should return
        def mock_itemdata_individual_signals():
            """Simulate ItemData.__getitem__ with individual_signals mode"""
            item_id = torch.tensor([5])
            text_features = torch.randn(768)  # Text embedding
            image_features = torch.randn(768)  # Image embedding
            
            # Return MultimodalSeqBatch for individual_signals mode
            return MultimodalSeqBatch(
                user_ids=-1 * torch.ones_like(item_id),
                ids=item_id,
                ids_fut=-1 * torch.ones_like(item_id),
                x_text=text_features,  # Separate text
                x_image=image_features,  # Separate image
                x_fut_text=-1 * torch.ones_like(text_features),
                x_fut_image=-1 * torch.ones_like(image_features),
                x_brand_id=torch.tensor([2]),
                x_fut_brand_id=torch.tensor([-1]),
                seq_mask=torch.ones_like(item_id, dtype=bool),
            )
        
        # Test the mock function
        item = mock_itemdata_individual_signals()
        
        assert is_multimodal_batch(item), "Should be multimodal batch"
        assert item.x_text.shape == torch.Size([768]), f"Text shape wrong: {item.x_text.shape}"
        assert item.x_image.shape == torch.Size([768]), f"Image shape wrong: {item.x_image.shape}"
        
        print(f"✅ Individual signals concept working")
        print(f"   - Text features: {item.x_text.shape}")
        print(f"   - Image features: {item.x_image.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Individual signals test failed: {e}")
        return False

def test_sequence_individual_signals():
    """Test individual_signals with sequence data"""
    print("\n2️⃣ Testing sequence individual_signals...")
    
    try:
        def mock_seqdata_individual_signals():
            """Simulate SeqData.__getitem__ with individual_signals mode"""
            seq_len = 8
            item_ids = torch.randint(0, 100, (seq_len,))
            item_ids_fut = torch.randint(0, 100, (1,))
            
            # Simulate getting features for sequence
            text_sequence = torch.randn(seq_len, 768)  # Text sequence
            image_sequence = torch.randn(seq_len, 768)  # Image sequence
            
            return MultimodalSeqBatch(
                user_ids=torch.tensor([10]),
                ids=item_ids,
                ids_fut=item_ids_fut,
                x_text=text_sequence,  # [seq_len, 768]
                x_image=image_sequence,  # [seq_len, 768]
                x_fut_text=torch.randn(1, 768),
                x_fut_image=torch.randn(1, 768),
                x_brand_id=torch.randint(0, 20, (seq_len,)),
                x_fut_brand_id=torch.randint(0, 20, (1,)),
                seq_mask=torch.ones(seq_len, dtype=bool),
            )
        
        # Test sequence
        seq = mock_seqdata_individual_signals()
        
        assert is_multimodal_batch(seq), "Should be multimodal batch"
        assert len(seq.x_text.shape) == 2, f"Text should be 2D: {seq.x_text.shape}"
        assert len(seq.x_image.shape) == 2, f"Image should be 2D: {seq.x_image.shape}"
        assert seq.x_text.shape[0] == seq.x_image.shape[0], "Text and image should have same seq length"
        
        print(f"✅ Sequence individual signals working")
        print(f"   - Text sequence: {seq.x_text.shape}")
        print(f"   - Image sequence: {seq.x_image.shape}")
        print(f"   - Valid positions: {seq.seq_mask.sum().item()}")
        return True
        
    except Exception as e:
        print(f"❌ Sequence test failed: {e}")
        return False

def test_collate_function():
    """Test multimodal collate function"""
    print("\n3️⃣ Testing collate function...")
    
    try:
        from data.utils import multimodal_collate_fn
        
        # Create list of individual items (simulating dataset samples)
        batch_list = []
        for i in range(3):
            item = MultimodalSeqBatch(
                user_ids=torch.tensor([i]),
                ids=torch.tensor([[i*10]]),
                ids_fut=torch.tensor([[i*10+1]]),
                x_text=torch.randn(1, 768),
                x_image=torch.randn(1, 768),
                x_fut_text=torch.randn(1, 768),
                x_fut_image=torch.randn(1, 768),
                x_brand_id=torch.tensor([[i]]),
                x_fut_brand_id=torch.tensor([[i+1]]),
                seq_mask=torch.ones(1, 1, dtype=bool),
            )
            batch_list.append(item)
        
        # Test collate
        collated = multimodal_collate_fn(batch_list)
        
        expected_batch_size = 3
        assert collated.x_text.shape[0] == expected_batch_size, f"Wrong batch size: {collated.x_text.shape}"
        assert is_multimodal_batch(collated), "Collated batch should be multimodal"
        
        print(f"✅ Collate function working")
        print(f"   - Collated text: {collated.x_text.shape}")
        print(f"   - Collated image: {collated.x_image.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Collate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_attention_readiness():
    """Test readiness for cross-attention"""
    print("\n4️⃣ Testing cross-attention readiness...")
    
    try:
        # Create multimodal sequence
        batch_size, seq_len = 2, 5
        text_features = torch.randn(batch_size, seq_len, 768)
        image_features = torch.randn(batch_size, seq_len, 768)
        
        print(f"✅ Input features ready:")
        print(f"   - Text: {text_features.shape}")
        print(f"   - Image: {image_features.shape}")
        
        # Simulate cross-attention processing
        hidden_dim = 64
        
        # Project to same dimension
        text_proj = torch.nn.Linear(768, hidden_dim)
        image_proj = torch.nn.Linear(768, hidden_dim)
        
        text_hidden = text_proj(text_features)  # [2, 5, 64]
        image_hidden = image_proj(image_features)  # [2, 5, 64]
        
        # Simulate attention
        attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        text_attended, _ = attention(text_hidden, image_hidden, image_hidden)
        
        # Combine
        fused = torch.cat([text_attended, image_hidden], dim=-1)  # [2, 5, 128]
        output_proj = torch.nn.Linear(128, hidden_dim)
        final = output_proj(fused)  # [2, 5, 64]
        
        print(f"✅ Cross-attention simulation successful:")
        print(f"   - Final output: {final.shape}")
        print(f"   - Ready for RQ-VAE!")
        return True
        
    except Exception as e:
        print(f"❌ Cross-attention test failed: {e}")
        return False

def main():
    """Run all Phase 2 tests"""
    print("🚀 Starting Phase 2 verification tests...")
    
    tests = [
        ("Individual Signals Concept", test_individual_signals_concept),
        ("Sequence Individual Signals", test_sequence_individual_signals), 
        ("Collate Function", test_collate_function),
        ("Cross-Attention Readiness", test_cross_attention_readiness),
    ]
    
    success_count = 0
    
    for test_name, test_func in tests:
        if test_func():
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"📊 PHASE 2 RESULTS: {success_count}/{len(tests)} tests passed")
    print(f"{'='*50}")
    
    if success_count == len(tests):
        print("🎉 PHASE 2 CONCEPT VERIFIED!")
        print("✅ individual_signals mode concept is solid")
        print("✅ MultimodalSeqBatch handling works")
        print("✅ Cross-attention pipeline ready")
        print("\n🚀 NEXT: Implement individual_signals in your actual ItemData/SeqData")
    else:
        print(f"⚠️ {len(tests) - success_count} tests failed")

if __name__ == "__main__":
    main()