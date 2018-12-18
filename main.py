#code:utf8
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
import models.bn_inception as network
import models.embed as embed
from sample_data.sample_data import Preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

def run():
	net = network.bn_inception(pretrained = True)
	embed.embed(net, sz_embedding=cfg.EMBEDDING_WIDTH,normalize_output = True)
	if cfg.USE_CUDA==1: 
		net.cuda()
	metric_loss = get_loss(n_input=cfg.N, k=cfg.K, tau=cfg.TAU,n_pos=cfg.POS_SAMPLE_NUM, margin=cfg.MARGIN,input_dim=cfg.EMBEDDING_WIDTH,output_dim=cfg.TRAIN_CLASS,batch_size=cfg.BATCH_SIZE,method=cfg.METHOD)
	softmax_loss = get_loss(input_dim=cfg.EMBEDDING_WIDTH,output_dim=cfg.TRAIN_CLASS,margin=cfg.SOFTMAX_MARGIN,method=cfg.SOFTMAX_METHOD)
	
	optimizer = torch.optim.Adam(
    [
        { # embedding parameters
            'params': net.embedding_layer.parameters(), 
            'lr' : cfg.EMBD_LR
        },
        { # softmax loss parameters
            'params': softmax_loss.parameters(), 
            'lr': cfg.SOFTMAX_LOSS_LR
        },
        { # inception parameters, excluding embedding layer
            'params': list(
                set(
                    net.parameters()
                ).difference(
                    set(net.embedding_layer.parameters())
                )
            ), 
            'lr' : cfg.NET_LR
        }
    ],
    eps = 1e-2,
    weight_decay = cfg.WEIGHT_DECAY
)
	
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 10, 16],gamma = cfg.GAMMA)
	
	if cfg.TRAINING_OLD==1:
		print("Load model params")
		net.load_state_dict(torch.load(cfg.MODEL_PATH+str(cfg.MARGIN)+str(cfg.EMBEDDING_WIDTH)+str(cfg.DATASET)+str(cfg.METRIC_LOSS_PARAM)+'x'+str(cfg.METHOD)+str(cfg.CE_LOSS_PARAM)+'x'+str(cfg.SOFTMAX_METHOD)+str(cfg.K)+str(cfg.POS_SAMPLE_NUM)+".pkl"))

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
	run_num=0
	sprarsity=0
	err_pos=0
	old_err_pos=0
	total_sprarsity=0
	total_err_pos=0

	for epoch in range(cfg.EPOCH):
	#train
		scheduler.step()
		totalEpochLoss=0
		if cfg.METHOD==0:
			iter_num=int(cfg.DATASET_NUM/(cfg.N+1)/cfg.BATCH_SIZE)
		if cfg.METHOD==1:
			iter_num=int(cfg.DATASET_NUM/cfg.BATCH_SIZE)	
		if cfg.METHOD==3 or cfg.METHOD==4 or cfg.METHOD==5 or cfg.METHOD==6 or cfg.METHOD==7 or cfg.METHOD==2:
			iter_num=int(cfg.DATASET_NUM/cfg.BATCH_SIZE/2)	
		for iter in tqdm(range(iter_num)):
			batch_img,y_,real_y=preprocess.next_train_batch(cfg.POS_SAMPLE_NUM,cfg.N, cfg.BATCH_SIZE)
			optimizer.zero_grad()
			if cfg.USE_CUDA==1:
				out=net(batch_img.cuda())
			else:
				out=net(batch_img)
#*****************************************************for prec@k loss******************************************#
			if cfg.METHOD==0:	
				if cfg.USE_CUDA==1:
					loss_metric,sprarsity,old_err_pos=metric_loss(out,y_.cuda())
					loss=cfg.METRIC_LOSS_PARAM*loss_metric+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(out,y_)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#*****************************************************for angular loss******************************************#
			if cfg.METHOD==4:
				embed1=out[0:cfg.BATCH_SIZE,:]
				embed2=out[cfg.BATCH_SIZE:2*cfg.BATCH_SIZE,:]
				if cfg.USE_CUDA==1:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#*****************************************************for cluster loss******************************************#
			if cfg.METHOD==2:
				embed1=out[0:cfg.BATCH_SIZE,:]
				embed2=out[cfg.BATCH_SIZE:2*cfg.BATCH_SIZE,:]
				if cfg.USE_CUDA==1:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(out,real_y.cuda())+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(out,real_y)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#*****************************************************for npair loss******************************************#
			if cfg.METHOD==3:
				embed1=out[0:cfg.BATCH_SIZE,:]
				embed2=out[cfg.BATCH_SIZE:2*cfg.BATCH_SIZE,:]
				if cfg.USE_CUDA==1:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#*****************************************************for lifted loss******************************************#
			if cfg.METHOD==5:
				if cfg.USE_CUDA==1:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(out,y_.cuda())+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(out,y_)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#*****************************************************for triplet loss******************************************#
			if cfg.METHOD==6:
				embed1=out[0:cfg.BATCH_SIZE,:]
				embed2=out[cfg.BATCH_SIZE:2*cfg.BATCH_SIZE,:]
				if cfg.USE_CUDA==1:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#*****************************************************for contrastive loss******************************************#
			if cfg.METHOD==7:
				embed1=out[0:cfg.BATCH_SIZE,:]
				embed2=out[cfg.BATCH_SIZE:2*cfg.BATCH_SIZE,:]
				if cfg.USE_CUDA==1:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2,y_.cuda())+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y.cuda())
				else:
					loss=cfg.METRIC_LOSS_PARAM*metric_loss(embed1,embed2,y_)+cfg.CE_LOSS_PARAM*softmax_loss(out,real_y)
#**********************************************************end*************************************************#
			err_pos=old_err_pos
			totalEpochLoss=totalEpochLoss+loss.data
			if math.isnan(loss.data)==False:
				loss.backward()
				optimizer.step()
			else:
				print(loss.data)
			#res=val.test_training_dataset(out,y_,5,cfg.BATCH_SIZE)
			total_sprarsity+=sprarsity/cfg.SHOW_PER_ITER
			total_err_pos+=err_pos/cfg.SHOW_PER_ITER
				
			if run_num%cfg.SHOW_PER_ITER==cfg.SHOW_PER_ITER-1:
				preck_test,recallk_test,_,_,_=val.test(net,preprocess.test_loader,cfg.TEST_K,metric)
				#preck_train,recallk_train,_,_,_=val.test(net,preprocess.test_train_loader,cfg.TEST_K,metric)
				print("iter:",run_num,"prec@K:",preck_test,"recall@K:",recallk_test)
				torch.save(net.state_dict(), cfg.MODEL_PATH+str(cfg.MARGIN)+str(cfg.EMBEDDING_WIDTH)+str(cfg.DATASET)+str(cfg.METRIC_LOSS_PARAM)+'x'+str(cfg.METHOD)+str(cfg.CE_LOSS_PARAM)+'x'+str(cfg.SOFTMAX_METHOD)+str(cfg.K)+str(cfg.POS_SAMPLE_NUM)+".pkl")	
				'''
				ratio=total_sprarsity/total_err_pos
				output = open('m'+str(cfg.MARGIN)+'k'+str(cfg.K)+'n+'+str(cfg.POS_SAMPLE_NUM)+'tau'+str(cfg.TAU)+str(cfg.DATASET)+'.txt', 'a')
				output.write(str(preck_test[0]))
				output.write(' ')
				output.write(str(preck_test[1]))
				output.write(' ')
				output.write(str(preck_test[2]))
				output.write(' ')
				output.write(str(preck_test[3]))
				output.write(' ')
				output.write(str(recallk_test[0]))
				output.write(' ')
				output.write(str(recallk_test[1]))
				output.write(' ')
				output.write(str(recallk_test[2]))
				output.write(' ')
				output.write(str(recallk_test[3]))
				
				output.write(str(preck_train[0]))
				output.write(' ')
				output.write(str(preck_train[1]))
				output.write(' ')
				output.write(str(preck_train[2]))
				output.write(' ')
				output.write(str(preck_train[3]))
				output.write(' ')
				output.write(str(total_sprarsity))
				output.write(' ')
				output.write(str(ratio))
				
				output.write('\r')
				output.close()
				total_sprarsity=0
				total_err_pos=0
				'''
			run_num+=1
		print("\r\nEpoch:",epoch,"tau:",cfg.TAU,"avgEpochLoss:",totalEpochLoss/iter_num)

if __name__ == '__main__':
	run()

