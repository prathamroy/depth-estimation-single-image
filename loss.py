import torch
import torch.nn.functional as F

def patch_ranking_loss(depth_map, patch_pairs):
    loss = 0
    for (p1, p2, label) in patch_pairs:
        d1 = depth_map[:, :, p1[1], p1[0]]
        d2 = depth_map[:, :, p2[1], p2[0]]
        sign = 1 if label == 'p1_closer' else -1
        loss += torch.log(1 + torch.exp(-sign * (d1 - d2)))
    return loss / len(patch_pairs)

def smoothness_loss(depth_map, image):
    dx = torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:])
    dy = torch.abs(depth_map[:, :, :-1, :] - depth_map[:, :, 1:, :])
    dx_img = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    dy_img = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return dx.mean() * torch.exp(-dx_img) + dy.mean() * torch.exp(-dy_img)