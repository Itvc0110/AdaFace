import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import config
from net import build_model
from convert import batch_preprocess_images
from data import get_inference_dataloader

def main():
    args = config.get_args()
    device = torch.device(args.device)
    
    # Build and load model
    model = build_model(args.arch)
    model.to(device)
    model.eval()
    
    # Load checkpoint 
    ckpt_path = args.checkpoint_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    raw_sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(raw_sd, dict) and 'state_dict' in raw_sd:
        raw_sd = raw_sd['state_dict']
    normalized_sd = {k.replace('module.', '').replace('model.', ''): v for k, v in raw_sd.items()}
    model_keys = set(model.state_dict().keys())
    filtered_sd = {k: normalized_sd[k] for k in normalized_sd if k in model_keys}
    model.load_state_dict(filtered_sd, strict=False)
    print(f"Loaded model {args.arch} from {ckpt_path} on {device}")
    
    # Get dataloader 
    dataloader = get_inference_dataloader(args.input_dir, args.batch_size, args.num_workers)
    print(f"Found {len(dataloader.dataset)} images in {args.input_dir}")
    
    # Inference loop
    start_time = time.time()
    all_embeddings = []
    all_paths = []
    for batch_paths in dataloader:
        inputs, successful_paths = batch_preprocess_images(batch_paths, args.swap_color_channel, device)
        if inputs is None:
            continue
        with torch.no_grad():
            outputs = model(inputs)
            feats = outputs[0] if isinstance(outputs, tuple) else outputs
            embeddings = F.normalize(feats, dim=1).cpu().numpy()
        all_embeddings.extend(embeddings)
        all_paths.extend(successful_paths)
    
    elapsed = time.time() - start_time
    print(f"Inference completed. Time taken: {elapsed:.2f} seconds for {len(all_paths)} images")
    
    # Save embeddings
    for emb, path in zip(all_embeddings, all_paths):
        filename = os.path.splitext(os.path.basename(path))[0] + '.npy'
        np.save(os.path.join(args.output_dir, filename), emb)
    print(f"Saved {len(all_paths)} embeddings to {args.output_dir}")

if __name__ == '__main__':
    main()