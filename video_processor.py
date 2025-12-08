import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from PIL import Image
from convert import get_aligned_face_repo, preprocess_bgr_112_from_aligned

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_video(args, model, device):
    """Process video with human tracking, face embedding, DB matching, and re-identification."""
    if not args.video_path or not os.path.exists(args.video_path):
        raise ValueError(f"Video not found: {args.video_path}")
    
    # Load DB
    with open(args.db_path, 'r') as f:
        db = json.load(f)
    db_embeddings = {name: np.array(emb) for name, emb in db.items() if emb}
    if not db_embeddings:
        raise ValueError("Empty DB - build it first with --build_db")
    
    # Load YOLOv8
    yolo = YOLO('yolov8n.pt')  # Use 'yolov8m.pt' for better accuracy if needed
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Re-ID structures
    known_persons = {}  # person_id (int) -> {'avg_emb': np.array or None, 'name': str or None}
    next_person_id = 0
    
    # Track state: track_id -> {'embeddings': deque, 'person_id': int or None, 'frame_count': 0}
    tracks = defaultdict(lambda: {'embeddings': deque(maxlen=args.max_embs_per_track), 'person_id': None, 'frame_count': 0})
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        orig_frame = frame.copy()  # For annotation
        
        # YOLO track persons
        results = yolo.track(frame, persist=True, classes=0, conf=args.conf_threshold, iou=args.iou_threshold)
        
        for result in results[0].boxes:
            if not result.id:  # No track ID
                continue
            track_id = int(result.id)
            bbox = result.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            if tracks[track_id]['person_id'] is None:
                tracks[track_id]['person_id'] = None  # Will assign later
            
            tracks[track_id]['frame_count'] += 1
            
            # Attempt embedding every skip_interval frames
            if tracks[track_id]['frame_count'] % args.skip_interval == 0:
                # Crop upper body/head with padding
                person_h = bbox[3] - bbox[1]
                person_w = bbox[2] - bbox[0]
                pad_h = person_h * args.padding_ratio
                pad_w = person_w * args.padding_ratio
                crop_y1 = max(0, int(bbox[1] - pad_h))
                crop_y2 = min(height, int(bbox[1] + person_h * args.upper_crop_ratio + pad_h))
                crop_x1 = max(0, int(bbox[0] - pad_w))
                crop_x2 = min(width, int(bbox[2] + pad_w))
                
                if (crop_y2 - crop_y1 < args.min_crop_size) or (crop_x2 - crop_x1 < args.min_crop_size):
                    print(f"Frame {frame_num}, Track {track_id}: Crop too small, skipping.")
                    continue
                
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                # MTCNN align
                aligned_pil = get_aligned_face_repo(rgb_pil_image=crop_pil)
                if aligned_pil is None:
                    print(f"Frame {frame_num}, Track {track_id}: No face detected in crop.")
                    continue
                
                # AdaFace embedding
                input_tensor = preprocess_bgr_112_from_aligned(aligned_pil, args.swap_color_channel, normalize=True, device=device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    feat = outputs[0] if isinstance(outputs, tuple) else outputs
                    emb = F.normalize(feat, dim=1).cpu().numpy().flatten()
                tracks[track_id]['embeddings'].append(emb)
                print(f"Frame {frame_num}, Track {track_id}: Added embedding (total {len(tracks[track_id]['embeddings'])})")
            
            # Match/Re-ID if enough embeddings
            if len(tracks[track_id]['embeddings']) >= args.min_embs_for_match:
                avg_emb = np.mean(tracks[track_id]['embeddings'], axis=0)
                
                # Find best DB match
                max_db_sim = -1
                best_name = None
                for name, db_emb in db_embeddings.items():
                    sim = cosine_similarity(avg_emb, db_emb)
                    if sim > max_db_sim:
                        max_db_sim = sim
                        best_name = name
                name = best_name if max_db_sim > args.cos_threshold else None
                if name:
                    print(f"Frame {frame_num}, Track {track_id}: DB match to {name} (sim={max_db_sim:.2f})")
                
                # Re-ID: Check if matches a known person
                matched_pid = None
                max_reid_sim = -1
                for pid, data in known_persons.items():
                    if data['avg_emb'] is not None:
                        sim = cosine_similarity(avg_emb, data['avg_emb'])
                        if sim > args.reid_threshold and sim > max_reid_sim:
                            # Prefer if names match or both unknown
                            if (data['name'] == name) or (data['name'] is None and name is None):
                                matched_pid = pid
                                max_reid_sim = sim
                                break
                            elif sim > max_reid_sim:  # Still consider others
                                matched_pid = pid
                                max_reid_sim = sim
                
                if matched_pid is not None:
                    # Merge to existing person
                    tracks[track_id]['person_id'] = matched_pid
                    # Update known avg_emb (simple average)
                    old_emb = known_persons[matched_pid]['avg_emb']
                    known_persons[matched_pid]['avg_emb'] = (old_emb + avg_emb) / 2
                    # Update name if new info
                    if name and known_persons[matched_pid]['name'] is None:
                        known_persons[matched_pid]['name'] = name
                    print(f"Frame {frame_num}, Track {track_id}: Re-ID matched to Person {matched_pid} (sim={max_reid_sim:.2f})")
                else:
                    # New person
                    person_id = next_person_id
                    next_person_id += 1
                    known_persons[person_id] = {'avg_emb': avg_emb, 'name': name}
                    tracks[track_id]['person_id'] = person_id
                    print(f"Frame {frame_num}, Track {track_id}: New Person {person_id}")
            
            # Annotate
            pid = tracks[track_id]['person_id']
            if pid is None:
                label = "No face detected"
                display_id = track_id  # Fallback to track_id if no person_id yet
            else:
                person_data = known_persons[pid]
                label = person_data['name'] if person_data['name'] else "Unknown"
                display_id = pid
            cv2.rectangle(orig_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(orig_frame, f"Person {display_id}: {label}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(orig_frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved to {args.output_video_path}")