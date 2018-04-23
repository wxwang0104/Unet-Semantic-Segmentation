import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Model(nn.Module):

    def forward(self, *input):
        pass

    def predict(self, img):
        """
        Given an image, predict the segmentation mask.
        :param img: An image, which is a torch tensor of size (b, c, h, w)
        :return: A segmentation, which is a numpy array of size (h, w)
        """
        img = Variable(img)
        seg = self.forward(img)
        seg = seg.cpu().data.numpy()
        seg = np.transpose(seg, (0, 2, 3, 1))
        seg = np.argmax(seg, axis=-1)
        return seg
