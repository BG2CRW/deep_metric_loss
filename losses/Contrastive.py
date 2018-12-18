import torch
import torch.nn as nn

def Contrastive(margin):
    return ContrastiveLoss(margin=margin)

class ContrastiveLoss(nn.Module):

    def __init__(self,margin,eps=1e-12):
        super(ContrastiveLoss, self).__init__()
        self.margin = 1.0
        self.eps=1e-6
    def forward(self, embedding1, embedding2, label):
        cnt = embedding1.size(0)
        dist_sqr=torch.sum((embedding1-embedding2+self.eps)**2, 1)
        dist=torch.sqrt(dist_sqr)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        #print(dist)
        loss = label.float() * dist_sqr + (1 - label.float()) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / cnt
        return loss