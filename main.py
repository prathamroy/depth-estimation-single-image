import torch
from torchvision import transforms
from PIL import Image
from model import DepthEstimator
from loss import patch_ranking_loss, smoothness_loss
from utils import generate_patch_pairs, show_and_save_depth_map, overlay_depth_on_image, show_depth_vs_rgb_edges

device = torch.device('cpu')

# Loading image
img = Image.open("image.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
image = transform(img).unsqueeze(0).to(device)
model = DepthEstimator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred_depth = model(image)
    patch_pairs = generate_patch_pairs(image)
    loss = patch_ranking_loss(pred_depth, patch_pairs) + 0.1 * smoothness_loss(pred_depth, image)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        show_and_save_depth_map(pred_depth, f"figures/depth_map_epoch{epoch}.png")

# Final outputs
show_and_save_depth_map(pred_depth, "figures/depth_map_final.png")
overlay_depth_on_image(image, pred_depth, "figures/overlay_final.png")
show_depth_vs_rgb_edges(image, pred_depth, "figures/edges_comparison.png")