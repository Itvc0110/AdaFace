import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as trans

# Add repo to sys.path if needed (assuming running from repo root)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from face_alignment import mtcnn as repo_mtcnn
except Exception as e:
    raise RuntimeError("Could not import face_alignment.mtcnn from the repo.") from e

# Instantiate MTCNN (uses npy weights from face_alignment/mtcnn_pytorch/src/weights)
mtcnn_model = repo_mtcnn.MTCNN(device='cuda:0' if torch.cuda.is_available() else 'cpu', crop_size=(112, 112)) # change later for device sync

def get_aligned_face_repo(image_path=None, rgb_pil_image=None, limit=1):
    """
    Use the repo MTCNN to return a PIL.Image (RGB, 112x112) of the aligned face or None.
    """
    if rgb_pil_image is None and image_path is None:
        raise ValueError("Provide image_path or rgb_pil_image.")
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        if not isinstance(rgb_pil_image, Image.Image):
            raise TypeError("rgb_pil_image must be a PIL.Image")
        img = rgb_pil_image.convert('RGB')
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        if faces and len(faces) > 0:
            return faces[0]
        return None
    except Exception as e:
        print("[get_aligned_face_repo] detection/alignment failed:", e)
        return None

def preprocess_bgr_112_from_aligned(aligned_pil, swap_color_channel=False, normalize=True, device='cpu'):
    """
    Convert aligned PIL RGB 112x112 -> torch tensor (1,3,112,112) in BGR order,
    normalized with (x-0.5)/0.5 if normalize=True.
    """
    if aligned_pil is None:
        return None
    arr = np.array(aligned_pil.convert('RGB')).astype(np.float32) / 255.0  # HWC RGB [0,1]
    if swap_color_channel:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        arr = arr[:, :, ::-1]  # RGB to BGR
    if normalize:
        arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1)).copy()  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device).float()  # (1,3,H,W)
    return tensor

def load_and_preprocess_image(image_path, swap_color_channel=False, device='cpu'):
    """
    Load image, align face using MTCNN, preprocess to normalized tensor (BGR order for AdaFace).
    Returns: torch.Tensor (1, 3, 112, 112) or None if alignment fails.
    """
    aligned_pil = get_aligned_face_repo(image_path=image_path)
    return preprocess_bgr_112_from_aligned(aligned_pil, swap_color_channel, normalize=True, device=device)

def batch_preprocess_images(image_paths, swap_color_channel=False, device='cpu'):
    """
    Preprocess a list of images into a batched tensor.
    Skips failed images, returns tensor and list of successful paths.
    """
    tensors = []
    successful_paths = []
    for path in image_paths:
        t = load_and_preprocess_image(path, swap_color_channel, device)
        if t is not None:
            tensors.append(t)
            successful_paths.append(path)
    if not tensors:
        return None, []
    return torch.cat(tensors, dim=0), successful_paths