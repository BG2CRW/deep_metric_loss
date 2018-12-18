import torch
import torch.autograd as ag

from losses.sigma.polynomial.sp import log_sum_exp, LogSumExp
from losses.sigma.logarithm import LogTensor

def Topk_Hard_SVM():
    def fun(x,k):
        max_1, _ = x.topk(k, dim=1)
        max_1 = max_1.sum(1)

        return max_1
    return fun

def Topk_Smooth_SVM(tau):    
    def fun(x,k):
        lsp = LogSumExp(k)
        x.div_(tau)

        res1 = lsp(x)
        term_1 = res1[1]  #sigma k
        term_1 = LogTensor(term_1)

        loss = tau * term_1.torch()
        
        return loss
    return fun

def Top1_Smooth_SVM(tau):
    def fun(x):
        return tau * log_sum_exp(x / tau)
    return fun