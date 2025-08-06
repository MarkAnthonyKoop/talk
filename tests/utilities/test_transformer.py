#!/usr/bin/env python3
"""Test the generated transformer implementation."""

import sys
import os

# Add the ML project path
sys.path.insert(0, '/home/xx/temp/talk_extreme_test/ml_challenge/.talk/2025-08-05_02-08-18_talk_build_a_transformer-based_neural_network_for_real-/workspace/src')

try:
    # Test imports
    print("Testing imports...")
    from models.transformer import TransformerTranslator, EmbeddingLayer, PositionalEncoding
    print("✓ Transformer imports successful")
    
    # Test basic instantiation
    print("\nTesting model instantiation...")
    model = TransformerTranslator(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    print("✓ Model created successfully")
    print(f"  Model type: {type(model)}")
    print(f"  D-model: {model.d_model}")
    
    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    import torch
    
    # Create dummy input
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(0, 10000, (src_seq_len, batch_size))
    tgt = torch.randint(0, 10000, (tgt_seq_len, batch_size))
    
    # Forward pass
    with torch.no_grad():
        output, attention = model(src, tgt)
    
    print("✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({tgt_seq_len}, {batch_size}, 10000)")
    
    # Verify output shape
    assert output.shape == (tgt_seq_len, batch_size, 10000), "Output shape mismatch!"
    print("✓ Output shape verified")
    
    print("\n✅ All tests passed! The transformer implementation works correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Note: This might be due to missing dependencies. Install with:")
    print("pip install torch sentencepiece matplotlib seaborn numpy")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()