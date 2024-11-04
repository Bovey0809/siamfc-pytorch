from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        """
        fast cross correlation

        在SiamFC模型中，z和x分别表示模板图像和搜索图像的特征映射。

        z: 模板图像的特征映射，通常是一个较小的图像块，用于表示目标物体的外观。
        x: 搜索图像的特征映射，通常是一个较大的图像块，用于在其中搜索目标物体。

        在前向传播过程中，z和x通过卷积操作进行快速交叉相关，以计算相似度得分图。

        out: 相似度得分图，表示模板图像与搜索图像之间的相似度。
        """
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
