import torch
#import config.config_CUB200 as cfg 
#import config.config_CUB200_2011 as cfg
#import config.config_CARS196 as cfg 
import config.config_ONLINE_PRODUCT as cfg 
import os
from losses.main import get_loss
import numpy as np
from tqdm import tqdm
from torch import nn
import math
import val
from sklearn.manifold import TSNE
from time import time
import matplotlib as mpl
from PIL import Image
mpl.use('Agg')
import matplotlib.pyplot as plt
from sample_data.sample_data import Preprocess
import models.bn_inception as network
import models.embed as embed

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

def plot_embedding(data, path):
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)

	fig = plt.figure(figsize=(80,60))
	ax = plt.subplot(111)
	plt.xticks([])
	plt.yticks([])
	ax.axis('off') 
	#for i in tqdm(range(int(data.shape[0]/3))):
	for i in tqdm(range(2990)):
		ax1 = fig.add_axes([data[i, 0], data[i, 1],0.02,0.02])
		img = Image.open(path[i])
		ax1.imshow(img)
		ax1.axis('off')   
	return fig

def run():
	net = network.bn_inception(pretrained = True)
	embed.embed(net, sz_embedding=cfg.EMBEDDING_WIDTH,normalize_output = True)
	if cfg.USE_CUDA==1: 
		net.cuda()
	print("Load model params")
	net.load_state_dict(torch.load(cfg.MODEL_PATH+str(cfg.MARGIN)+str(cfg.EMBEDDING_WIDTH)+str(cfg.DATASET)+str(cfg.METRIC_LOSS_PARAM)+'x'+str(cfg.METHOD)+str(cfg.CE_LOSS_PARAM)+'x'+str(cfg.SOFTMAX_METHOD)+str(cfg.K)+str(cfg.POS_SAMPLE_NUM)+".pkl"))

	print("Index all dataset")
	preprocess = Preprocess(root=cfg.DATA_ROOT,use_cuda=cfg.USE_CUDA,test_batch_size=cfg.TEST_BATCH_SIZE,method=cfg.METHOD,dataset_name=cfg.DATASET,with_bounding_box=cfg.WITH_BOUNDING_BOX,download=cfg.DOWNLOAD)
	print("Done!")
	if cfg.METHOD==0:
		metric = 'cosine'
	else:
		metric = 'euclidean'
	print("embd_size=",cfg.EMBEDDING_WIDTH,"dataset=",cfg.DATASET)

	if cfg.METHOD==0:
		print("tau=",cfg.TAU,"K=",cfg.K,"N=",cfg.N,"N+=",cfg.POS_SAMPLE_NUM,"embd_width=",cfg.EMBEDDING_WIDTH,"batch_size=",cfg.BATCH_SIZE,'margin=',cfg.MARGIN)
	print("softmax_rate:",cfg.CE_LOSS_PARAM,"metric_rate:",cfg.METRIC_LOSS_PARAM)
	
	preck_test,recallk_test,X1,T1,path1=val.test(net,preprocess.test_loader,cfg.TEST_K,metric)
	preck_train,recallk_train,X2,T2,path2=val.test(net,preprocess.test_train_loader,cfg.TEST_K,metric)
	
	data=X1.cpu().numpy()
	label=T1.cpu().numpy()
	n_samples=len(path1)
	n_features=64
	X2.cpu().numpy()

	print('recall@1 in test set: ',str(recallk_test[0]))
	print('recall@1 in train set: ',str(recallk_train[0]))
	
	print('Computing t-SNE embedding')
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	
	result = tsne.fit_transform(data)

	fig = plot_embedding(result, path1)
	fig.savefig("str(cfg.DATASET)+tsne.jpg")
	#plt.show(fig)

if __name__ == '__main__':
	run()
