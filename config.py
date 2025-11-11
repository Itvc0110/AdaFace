import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='AdaFace Inference')
    parser.add_argument('--arch', default='ir_101', type=str, help='Backbone architecture (e.g., ir_101, ir_50)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/adaface_ir101_ms1mv3.pth', help='Path to the checkpoint file')
    parser.add_argument('--input_dir', type=str, default='./input_images', help='Directory containing input images for inference')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save embeddings')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--swap_color_channel', action='store_true', help='Swap color channels during preprocessing')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args