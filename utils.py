import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def generate_patch_pairs(image, num_pairs=100):
    pairs = []
    H, W = image.shape[2:]
    for _ in range(num_pairs):
        x1, y1 = random.randint(0, W-1), random.randint(0, H-1)
        x2, y2 = random.randint(0, W-1), random.randint(0, H-1)
        label = 'p1_closer' if y1 > y2 else 'p2_closer'
        pairs.append(((x1, y1), (x2, y2), label))
    return pairs

def show_and_save_depth_map(depth_map, save_path="figures/depth_map_final.png"):
    depth_np = depth_map.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_np, cmap='inferno')
    plt.axis('off')
    plt.title("Predicted Relative Depth Map")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def overlay_depth_on_image(image_tensor, depth_map, save_path="figures/overlay_final.png"):
    image = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    depth = depth_map.squeeze().detach().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(depth, cmap='inferno', alpha=0.6)
    plt.axis('off')
    plt.title("Overlay of RGB and Predicted Depth")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def show_depth_vs_rgb_edges(image_tensor, depth_map, save_path="figures/edges_comparison.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image_gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    depth_np = depth_map.squeeze().detach().cpu().numpy()

    edges_img = cv2.Canny(image_gray, 100, 200)
    edges_depth = cv2.Canny((depth_np * 255).astype(np.uint8), 100, 200)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(edges_img, cmap='gray')
    axs[0].set_title("Canny Edges - Original Image")
    axs[1].imshow(edges_depth, cmap='gray')
    axs[1].set_title("Canny Edges - Depth Map")
    for ax in axs:
        ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()