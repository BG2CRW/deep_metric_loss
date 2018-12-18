import torch
import torch.nn as nn
import numpy as np

def NPair(margin):
    #https://github.com/ronekko/deep_metric_learning/blob/master/lib/functions/n_pair_mc_loss.py
    return NPairLoss(margin=margin)

class NPairLoss(nn.Module):

    def __init__(self,margin):
        super(NPairLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.margin = margin

    def forward(self, embed_anchor, embed_pos):
        embed_anchor_norm = embed_anchor.norm(dim=1)
        embed_pos_norm = embed_pos.norm(dim=1)

        simliarity_matrix = embed_anchor.mm(embed_pos.transpose(0, 1))
        N=embed_anchor.size()[0]
        target = torch.from_numpy(np.array([i for i in range(N)])).cuda()
        l2loss = (embed_anchor_norm.sum()+embed_pos_norm.sum())/N
        return self.criterion(simliarity_matrix,target)+l2loss*0.001