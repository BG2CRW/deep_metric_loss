import torch
import torch.nn as nn

def Lifted(margin):
    #https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4
    return LiftedLoss(margin=margin)

class LiftedLoss(nn.Module):

    def __init__(self,margin):
        super(LiftedLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding, label):
        loss = 0
        counter = 0
        bsz = embedding.size(0)
        mag = (embedding ** 2).sum(1).expand(bsz, bsz)
        sim = embedding.mm(embedding.transpose(0, 1))  #simliarity inner product
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        for i in range(bsz):
            t_i = label[i]
            for j in range(i + 1, bsz):
                t_j = label[j]
                if t_i == t_j:
                    # Negative component
                    l_ni = (self.margin - dist[i][label != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][label != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    # Positive component
                    l_p  = dist[i,j]
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2  #max(x,0)
                    counter += 1
        return loss / (2 * counter)