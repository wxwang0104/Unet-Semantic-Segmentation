from utils import *
from dataset.transform import *

import imageio
from torch.utils.data import Dataset
from torchvision import transforms
import os

from skimage import transform
import cv2

class SemanticSegmentationDataset(Dataset):

    def __init__(self, data_dir, image_set, crop_size, mode='train'):
        """
        Initialization.
        :param data_dir: The folder root of the image samples.
        :param image_set: The image set file location, relative to 'root'.
        :param crop_size: The image and segmentation size after scale and crop for training.
        :param mode: 'train', 'valid', 'test'
        """
        self.data_dir = data_dir
        self.size = crop_size
        self.mode = mode

        # mean and std using ImageNet mean and std,
        # as required by http://pytorch.org/docs/master/torchvision/models.html
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        image_set = os.path.join(data_dir, image_set)
        self.image_locations = [x.strip('\n') for x in open(image_set, 'r').readlines() if x is not '\n']
        self.len = len(self.image_locations)
        self.img_paths = [os.path.join(data_dir, 'stage1_images', loc + '.png') for loc in self.image_locations]
        if self.mode in ['train', 'valid']:
            self.seg_paths = [os.path.join(data_dir, 'stage1_images', 'single_masks', loc.split('/')[1] + '.png')
                              for loc in self.image_locations]

    def __getitem__(self, index):
        """
        Get image from file.
        :param index: The index of the image.
        :return: The image, segmentation, and the image base name.
        """
        img_key = self.image_locations[index].split('/')[1].strip('.png')
        img_path = self.img_paths[index]
        # img = imageio.imread(img_path)[:, :, :3]  # remove alpha channel
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, :3] # remove alpha channel
        seg = np.full(img.shape[:2], -1)
        if self.mode in ['train', 'valid']:
            seg_path = self.seg_paths[index]
            seg = (imageio.imread(seg_path) > 128).astype(np.int)
            if self.mode in ['train']:
                img, seg = self.train_augment(img, seg)
            else:
                img, seg = self.valid_augment(img, seg)
            seg = seg.astype(np.long)
            seg = torch.from_numpy(seg.copy())
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).contiguous()
        return img, seg, img_key

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return self.len

    def train_augment(self, img, seg):
        img, seg = random_shift_scale_rotate_transform2(img, seg,
                                                        shift_limit=[0, 0], scale_limit=[1 / 2, 2],
                                                        rotate_limit=[-45, 45],
                                                        borderMode=cv2.BORDER_REFLECT_101,
                                                        u=0.5)  # borderMode=cv2.BORDER_CONSTANT

        # overlay = multi_mask_to_contour_overlay(seg, img, color=(0, 255, 0))
        # cv2.imwrite('results/overlay.png', overlay)

        img, seg = random_crop_transform2(img, seg, self.size, self.size, u=0.5)
        img, seg = random_horizontal_flip_transform2(img, seg, 0.5)
        img, seg = random_vertical_flip_transform2(img, seg, 0.5)
        img, seg = random_rotate90_transform2(img, seg, 0.5)

        return img, seg

    def valid_augment(self,  img, seg):
        img, seg = fix_crop_transform2(img, seg, -1, -1, self.size, self.size)
        return img, seg

    def train_augment_simple(self, img, seg):
        img, seg = self._scale_and_crop(img, seg, self.size, mode='train')
        if random.choice([-1, 1]) > 0:
            img, seg = self._flip(img, seg)
        return img, seg

    def valid_augment_simple(self, img, seg):
        img, seg = self._scale_and_crop(img, seg, self.size, mode='valid')
        return img, seg

    @staticmethod
    def _scale_and_crop(img, seg, crop_size, mode):
        """
        scale crop the image to make every image of the same square size, H = W = crop_size
        :param img: The image.
        :param seg: The segmentation of the image.
        :param crop_size: The crop size.
        :return: The cropped image and segmentation.
        """
        h, w = img.shape[0], img.shape[1]
        # if train:
        #     # random scale
        #     scale = random.random() + 0.5  # 0.5-1.5
        #     scale = max(scale, 1. * crop_size / (min(h, w) - 1))  # ??
        # else:
        #     # scale to crop size
        #     scale = 1. * crop_size / (min(h, w) - 1)
        scale = crop_size / min(h, w)
        if scale > 1:
            print('scale: ', scale)
            img = transform.rescale(img, scale, mode='reflect', order=1)  # order 1 is bilinear
            seg = transform.rescale(seg.astype(np.float), scale, mode='reflect', order=0)  # order 0 is nearest neighbor

        h_s, w_s = img.shape[0], seg.shape[1]
        if mode in ['train']:
            x1 = random.randint(0, w_s - crop_size)
            y1 = random.randint(0, h_s - crop_size)
        else:
            x1 = (w_s - crop_size) // 2
            y1 = (h_s - crop_size) // 2

        img_crop = img[y1: y1 + crop_size, x1: x1 + crop_size, :]
        seg_crop = seg[y1: y1 + crop_size, x1: x1 + crop_size]
        return img_crop, seg_crop

    @staticmethod
    def _flip(img, seg):
        img_flip = img[:, ::-1, :]
        seg_flip = seg[:, ::-1]
        return img_flip, seg_flip
