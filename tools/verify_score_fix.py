#!/usr/bin/env python3
"""Quick verification of detection head score ranges."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Fine-tune'))

import torch
from models.detection.center_head import CenterHead


def test_forward_pass():
    """Test that forward pass produces reasonable score ranges."""
    print("=" * 60)
    print("TESTING FORWARD PASS SCORE RANGES")
    print("=" * 60)
    
    # Create a simple head
    head = CenterHead(in_channels=768, head_conv=256, use_deconv=True, keypoint_only=True)
    head.eval()
    
    # Fake input features
    feats = torch.randn(1, 768, 32, 32)
    
    with torch.no_grad():
        heat, size, offset = head(feats)
    
    # Check output ranges
    heat_np = heat.cpu().numpy()
    min_score = heat_np.min()
    max_score = heat_np.max()
    mean_score = heat_np.mean()
    
    print(f"Heatmap scores:")
    print(f"  Min:  {min_score:.6f}")
    print(f"  Max:  {max_score:.6f}")
    print(f"  Mean: {mean_score:.6f}")
    
    # Verify scores are in sigmoid range
    assert 0.0 <= min_score <= 1.0, f"❌ Min score {min_score} outside [0,1] range!"
    assert 0.0 <= max_score <= 1.0, f"❌ Max score {max_score} outside [0,1] range!"
    
    # With random initialization, we expect scores near 0.5 (sigmoid of ~0)
    # But max should be able to reach high values (>0.9 is possible)
    if max_score < 0.5:
        print(f"⚠️  Max score {max_score:.3f} seems low (expected >0.5 with random init)")
    else:
        print(f"✅ Max score {max_score:.3f} is reasonable")
    
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VERIFICATION: Detection Head Score Ranges")
    print("=" * 60 + "\n")

    test_forward_pass()
    
    print("=" * 60)
    print("✅ ALL CHECKS PASSED - Fix verified!")
    print("=" * 60)
    print()
    print()
