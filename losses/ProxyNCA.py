import torch
import torch.nn as nn

def ProxyNCA(sz_embed, nb_classes):
    #https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py
    return ProxyNCALoss(sz_embed=sz_embed, nb_classes=nb_classes)

class ProxyNCALoss(nn.Module):

    def __init__(self, sz_embed, nb_classes):
        super(ProxyNCALoss,self).__init__()
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.proxies = torch.nn.Embedding(nb_classes, sz_embed).cuda()
        torch.nn.init.xavier_uniform_(self.proxies.weight)
        self.dist = lambda x, y : torch.pow(torch.nn.PairwiseDistance(eps = 1e-16)(x, y),2)
        # self.dist = torch.nn.PairwiseDistance(eps = 1e-16)

    def nca(self, xs, ys, i):
        # NOTE possibly something wrong with labels/classes ...
        # especially, if trained on labels in range 100 to 200 ... 
        # then y = ys[i] can be for example 105, but Z has 0 to 100
        # therefore, all proxies become negativ!
        x = xs[i] # embedding of sample i, produced by embedded bninception
        y = ys[i].long() # label of sample i
        # for Z: of all labels, select those unequal to label y
        Z = torch.masked_select( 
            torch.autograd.Variable(
                torch.arange(0, self.nb_classes).long()
            ).cuda(),
            torch.autograd.Variable(
                torch.arange(0, self.nb_classes).long()
            ).cuda() != y
        ).long()

        # all classes/proxies except of y
        assert Z.size(0) == self.nb_classes - 1 
        
        # with proxies embedding, select proxy i for target, p(ys)[i] <=> p(y)
        p_dist = torch.exp(
            - self.dist(
                torch.nn.functional.normalize(
                    self.proxies(y), # [1, batch_size], normalize along dim = 1 (batch_size)
                    dim = 0
                ),
                x.unsqueeze(0)
            )
        )      
        n_dist = torch.exp(
            - self.dist(
                torch.nn.functional.normalize(
                    self.proxies(Z), # [nb_classes - 1, batch_size]
                    dim = 1
                ),
                x.expand(Z.size(0), x.size(0)) # [nb_classes - 1, batch_size]
            )
        )
        return -torch.log(p_dist / torch.sum(n_dist))
    def forward(self, xs, ys):
        sz_batch=ys.size()[0]
        return torch.mean(
            torch.stack(
                [self.nca(xs, ys, i) for i in range(sz_batch)]
            )
        )