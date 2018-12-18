import sklearn.metrics.pairwise
import numpy as np
import torch

def predict_batchwise(model, dataloader):
    with torch.no_grad():
        X, Y, Z = zip(*[
            [x, y, z] for X, Y, Z in dataloader
                for x, y, z in zip(
                    model(X.cuda()).cpu(), 
                    Y, Z
                )
        ])
    return torch.stack(X), torch.stack(Y), Z

def assign_by_dist_at_k(X, T, k ,metric):
    """ 
    X : [nb_samples x nb_features], e.g. 3000 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = sklearn.metrics.pairwise.pairwise_distances(X,metric=metric)
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1]
    return np.array([[T[i] for i in ii] for ii in indices])

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    T=T.numpy()
    s=0
    for i in range(len(T)):
        if T[i] in Y[i,:][:k]:
            s += 1
    return s / (1. * len(T))

def calc_prec_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    temp=np.zeros((len(T),k))
    for i in range(len(T)):
        for j in range(k):
            if Y[i][j]==T[i]:
                temp[i][j]=1
            else:
                temp[i][j]=0
    s = sum(temp.sum(1)/k)
    return s / (1. * len(T))

def test(net,test_loader,test_k,metric):
        net_is_training = net.training
        net.eval()

        # calculate embeddings with model, also get labels (non-batch-wise)
        X, T, img_path = predict_batchwise(net, test_loader)
        # get predictions by assigning nearest test_k neighbors with euclidian
        Y = assign_by_dist_at_k(X, T, test_k,metric)
        # calculate recall @ 1, 3, 5, 10
        recall = []
        prec = []
        for k in [1, 3, 5, 10]:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            p_at_k = calc_prec_at_k(T, Y, k)
            prec.append(p_at_k)
        net.train(net_is_training) # revert to previous training state
        return prec,recall,X,T,img_path

def get_1d_array(N,i):
    mask = torch.zeros(N,requires_grad=False)
    j = 0
    counter = 0
    while counter!=N:
        if j==i:
            j+=1
        mask[counter]=j
        counter+=1
        j+=1
    return mask

def test_training_dataset(x,y,k,batch_size):
    prec=0.5
    embd_size = x.size()[1]
    n_pos=int(y[0].sum()+1)
    N=y.size()[1]
    for i in range(batch_size):
        for j in range(n_pos):
            if  j==0 and i==0:
                score=torch.mm(x[0,:].view(-1,embd_size),x[0:N+1,:].t())/torch.sqrt((x[0,:].view(-1,embd_size)**2).sum())/(x[0:N+1,:]**2).sum(1)
                y_=y[0].view(-1,N)
            else:               
                score_temp=torch.mm(x[j+i*(N+1),:].view(-1,embd_size),x[0+i*(N+1):N+1+i*(N+1),:].t())/torch.sqrt((x[j+i*(N+1),:].view(-1,embd_size)**2).sum())/(x[i*(N+1):i*(N+1)+1+N,:]**2).sum(1)
                score=torch.cat([score,score_temp],0)
                y_=torch.cat([y_,y_[0].view(-1,N)],0)
        
        for i in range(n_pos):
            if i==0:
                mask = get_1d_array(N,i).view(-1,N)
            else:
                mask=torch.cat([mask,get_1d_array(N,i).view(-1,N)],0)
        for i in range(batch_size):
            if i==0:
                new_mask=mask
            else:
                new_mask=torch.cat([new_mask,mask],0)
    score = score.gather(1, new_mask.cuda().long())
    index=score.topk(k)[1]
    res=torch.ge(index,n_pos-1).sum().float()/(index.size()[0]*k)
    return res