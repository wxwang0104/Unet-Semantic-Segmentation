
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight += filt
    return torch.from_numpy(weight).float()


def image_show(name, image, resize=1):
    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize * W), round(resize * H))


def multi_mask_to_color_overlay(multi_mask, image=None, color=None):
    height, width = multi_mask.shape[:2]
    overlay = np.zeros((height, width, 3), np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks == 0:
        return overlay

    if type(color) in [str] or color is None:
        # https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None:
            color = 'summer'  # 'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0, 1, 1 / num_masks))
        color = np.array(color[:, :3]) * 255
        color = np.fliplr(color)
        # np.random.shuffle(color)

    elif type(color) in [list, tuple]:
        color = [color for _ in range(num_masks)]

    for i in range(num_masks):
        mask = multi_mask == i + 1
        overlay[mask] = color[i]
        # overlay = instance[:,:,np.newaxis]*np.array( color[i] ) +  (1-instance[:,:,np.newaxis])*overlay

    return overlay


def multi_mask_to_contour_overlay(multi_mask, image=None, color=(255, 255, 255)):
    height, width = multi_mask.shape[:2]
    overlay = np.zeros((height, width, 3), np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks == 0:
        return overlay

    for i in range(num_masks):
        mask = multi_mask == i + 1
        contour = mask_to_inner_contour(mask)
        overlay[contour] = color

    return overlay


def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
            | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
            | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
            | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour


def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
            | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
            | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
            | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour
