import torch
import torch.nn as nn

def Triplet(margin):
    return TripletLoss(margin=margin)

class TripletLoss(nn.Module):

    def __init__(self,margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet=nn.TripletMarginLoss(margin=margin, p=2)
        self.dist = lambda x, y : torch.pow(torch.nn.PairwiseDistance(eps = 1e-16)(x, y),2)

    def forward(self, embed1,embed2):
        new_anchor=torch.tensor([]).cuda().float()
        new_pos=torch.tensor([]).cuda().float()
        new_neg=torch.tensor([]).cuda().float()
        for i in range(embed1.size()[0]):
            for j in range(embed1.size()[0]):
                if j!=i: #ap<an
                    #print(self.dist(embed1[i].view(1,-1),embed2[i].view(1,-1)))
                    #print(self.dist(embed1[i].view(1,-1),embed1[j].view(1,-1)))
                    if self.dist(embed1[i].view(1,-1),embed2[i].view(1,-1))<self.dist(embed1[i].view(1,-1),embed1[j].view(1,-1)):
                        new_anchor=torch.cat([new_anchor,embed1[i].view(1,-1)],dim=0)
                        new_pos=torch.cat([new_pos,embed2[i].view(1,-1)],dim=0)
                        new_neg=torch.cat([new_neg,embed1[j].view(1,-1)],dim=0)

                    if self.dist(embed1[i].view(1,-1),embed2[i].view(1,-1))<self.dist(embed1[i].view(1,-1),embed2[j].view(1,-1)):
                        new_anchor=torch.cat([new_anchor,embed1[i].view(1,-1)],dim=0)
                        new_pos=torch.cat([new_pos,embed2[i].view(1,-1)],dim=0)
                        new_neg=torch.cat([new_neg,embed2[j].view(1,-1)],dim=0)

                    if self.dist(embed2[i].view(1,-1),embed1[i].view(1,-1))<self.dist(embed2[i].view(1,-1),embed1[j].view(1,-1)):
                        new_anchor=torch.cat([new_anchor,embed2[i].view(1,-1)],dim=0)
                        new_pos=torch.cat([new_pos,embed1[i].view(1,-1)],dim=0)
                        new_neg=torch.cat([new_neg,embed1[j].view(1,-1)],dim=0)

                    if self.dist(embed2[i].view(1,-1),embed1[i].view(1,-1))<self.dist(embed2[i].view(1,-1),embed2[j].view(1,-1)):  
                        new_anchor=torch.cat([new_anchor,embed2[i].view(1,-1)],dim=0)
                        new_pos=torch.cat([new_pos,embed1[i].view(1,-1)],dim=0)
                        new_neg=torch.cat([new_neg,embed2[j].view(1,-1)],dim=0)
            
        loss=self.triplet(new_anchor,new_pos,new_neg)
        return loss