from __future__ import absolute_import

import torch
from torch import nn


class ClusterLoss(nn.Module):
    """ClusterLoss loss.

    Reference:

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(ClusterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, feat):
        #feat代表Zi,self.centers代表Wj
        #首先计算Z和W之间的欧式距离

        batch_size = feat.size(0)
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat, self.centers.t())#shape(batchsize,self.num_classes)
        #计算(Z和W间欧氏距离+1)的倒数矩阵
        distmat = 1.0/(distmat+1.0)
        D = distmat/distmat.sum(dim = 1,keepdim=True)#shape(batchsize,self.num_classes)
        Fj = D.sum(dim=0,keepdim =True)#shape(1,self.num_classes)
        Q_low = torch.pow(D,2)/Fj #shape(batchsize,self.num_classes)
        Q = Q_low / Q_low.sum(dim=1,keepdim=True) #shape(batchsize,self.num_classes)
        loss = D.mul(torch.log(D/Q)).sum() / batch_size#clamp(min=1e-12, max=1e+12)
        return loss

if __name__ == '__main__':
    use_gpu = True
    torch.manual_seed(0)
    center_loss = ClusterLoss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(2, 3).cuda()
        targets = torch.Tensor([0, 1]).long().cuda()

    loss = center_loss(features)
    print(loss)