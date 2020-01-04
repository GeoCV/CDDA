from __future__ import absolute_import

import torch
from torch import nn
from scipy.spatial.distance import cdist


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        a = x.max()
        b = x.min()
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        #得到每个x[0]和centers[0]特征间的欧氏距离，这样写是可能是为了便于矩阵运算
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()
        return loss

class ClusterLoss(nn.Module):
    """ClusterLoss loss.

    Reference:

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(ClusterLoss, self).__init__()
        self.num_classes = 2#num_classes
        self.feat_dim = 3#feat_dim
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
        loss = D.mul(torch.log(D/Q)).clamp(min=1e-12, max=1e+12).sum() / batch_size
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
