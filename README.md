# AdaFace Inference Repo

AdaFace for inference only.
Loads pre-trained model, aligns faces using MTCNN, computes embeddings.

## Setup
1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Place input images in `./input_images/` (or specify --input_dir).

## Usage
```bash
python main.py --arch ir_101 --checkpoint_path ./checkpoints/adaface_ir101_ms1mv3.pth --input_dir ./input_images --output_dir ./results