# Relative Depth Estimation from a Single Outdoor Image

This is a mini project for estimating relative depth from a single RGB image using self-supervised learning. 
- Only one image was used for training.
- The model learns relative depth by comparing random patches.

## Folder Structure
- `main.py`: Training code
- `model.py`: CNN model (encoder-decoder)
- `loss.py`: Loss functions (ranking and smoothness)
- `utils.py`: Patch sampling and visualizations
- `figures/`: Output depth maps and overlays
- `image.jpg`: The input image (resized to 512x512)

## Steps to Run
1. Setting up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Installing the required libraries:
    ```bash
    pip install torch torchvision matplotlib opencv-python
    ```
3. Run:
    ```bash
    python main.py
    ```

The output images will be saved in the `figures/` folder automatically.

## Acknowledgements
Inspired by ideas from SinGAN: Learning a Generative Model from a Single Natural Image, and self-supervised depth estimation papers.

Image credit: The Australian National University.
