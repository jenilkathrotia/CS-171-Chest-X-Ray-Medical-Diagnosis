import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def inspect_weights(filename):
    file_path = os.path.join(current_dir, filename)
    print(f"--- Inspecting: {file_path} ---")
    try:
        # Load the checkpoint
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # Print what metadata Jenil saved
        print(f"Top-level keys found: {list(checkpoint.keys())}")
        
        # Extract just the weights
        if 'state_dict' in checkpoint:
            print("\nFound 'state_dict'! Here are the first 5 layers:")
            weights = checkpoint['state_dict']
            keys = list(weights.keys())[:5]
            for key in keys:
                print(f" - Layer: {key} | Shape: {weights[key].shape}")
        else:
            print("Just a standard dict, no nested state_dict.")
            
    except Exception as e:
        print(f"Error loading file: {e}")
    print("\n")

inspect_weights('results/checkpoints/custom_cnn_best.pt')
inspect_weights('results/checkpoints/densenet121_best.pt')