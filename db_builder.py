import os
import numpy as np
import json
from PIL import Image
import torch
import torch.nn.functional as F
from convert import get_aligned_face_repo, preprocess_bgr_112_from_aligned

def build_employee_db(args, model, device):
    if not args.employees_dir or not os.path.isdir(args.employees_dir):
        raise ValueError(f"Employees directory not found: {args.employees_dir}")
    
    db = {}
    for employee_name in os.listdir(args.employees_dir):
        employee_dir = os.path.join(args.employees_dir, employee_name)
        if not os.path.isdir(employee_dir):
            continue
        image_paths = [os.path.join(employee_dir, f) for f in os.listdir(employee_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_paths:
            print(f"No images for {employee_name}, skipping.")
            db[employee_name] = []
            continue
        
        # Extract embeddings
        embeddings = []
        for path in image_paths:
            aligned_pil = get_aligned_face_repo(image_path=path)
            if aligned_pil is None:
                print(f"Face alignment failed for {path}, skipping.")
                continue
            input_tensor = preprocess_bgr_112_from_aligned(aligned_pil, args.swap_color_channel, normalize=True, device=device)
            with torch.no_grad():
                outputs = model(input_tensor)
                feat = outputs[0] if isinstance(outputs, tuple) else outputs
                emb = F.normalize(feat, dim=1).cpu().numpy().flatten()
            embeddings.append(emb)
        
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0).tolist()
            db[employee_name] = avg_emb
            print(f"Built embedding for {employee_name} from {len(embeddings)} images.")
        else:
            db[employee_name] = []
    
    with open(args.db_path, 'w') as f:
        json.dump(db, f)
    print(f"Saved DB to {args.db_path}")