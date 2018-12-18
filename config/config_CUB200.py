DATA_ROOT="/export/home/datasets"
DATASET = 'CUB200' #0:CIFAR100  1:CUB200   2:CARS196
EMBEDDING_WIDTH =64
DATASET_NUM = 2990   #train
WITH_BOUNDING_BOX = False
DOWNLOAD = False
TEST_K = 10
TEST_BATCH_SIZE = 10
#general parameter
USE_CUDA = 1
EPOCH = 500
LAMDA=1
METRIC_LOSS_PARAM = LAMDA
CE_LOSS_PARAM = 1-LAMDA
TRAINING_OLD = 0
TRAIN_CLASS = 100    #cifar100:80 CUB200:100 CARS196:98

MODEL_PATH = "model_param/preck_loss"
K = 5            #3 5  10
N = 55             #
POS_SAMPLE_NUM = 15  #8 10  20
TAU = 0.001  #0.1           #10->2   

GPU_NUM = '1'
METHOD = 0 #0:smooth prec@k loss  2:clustering loss  3:npair loss  4:angular loss  
		   #5:lifted loss 6:triplet loss 7:contrastive loss

if METHOD == 0:      #0:smooth prec@k loss 
	SOFTMAX_METHOD = 10   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 0.4     #up to 0.4 
	SOFTMAX_MARGIN = 0.5   #L-Softmax:2  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 13
	BATCH_SIZE = 2   #0:20     1:200     2:500     3:500
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3   #1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=0.12   #0.15


if METHOD == 3:      #npair loss
	SOFTMAX_METHOD = 12   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 0.4       #up to 0.4 
	SOFTMAX_MARGIN = 1   #L-Softmax:2  A-Softmax:4  LMCL:1  arcface:0.5
	SHOW_PER_ITER = 13
	BATCH_SIZE = 60   #0:20     1:200     2:500     3:500
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-3
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=0#1e-1

if METHOD == 4:#angular loss
	SOFTMAX_METHOD = 10   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 45       #45degree
	SOFTMAX_MARGIN = 4   #L-Softmax:2  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 17
	BATCH_SIZE = 45   #0:20     1:200     2:500     3:500
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-4
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 2:  #clustering loss 
	MARGIN = 1  
	SOFTMAX_METHOD = 14   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 3    #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 34
	BATCH_SIZE = 64    #<98
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1


if METHOD == 5:      #lifted loss
	MARGIN = 0.6      #lifted:2
	SOFTMAX_METHOD = 11   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 3   #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 15
	BATCH_SIZE = 100
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-4
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1


if METHOD == 6:      #triplet loss
	MARGIN = 0.5
	SOFTMAX_METHOD = 14   #10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 2      #  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SHOW_PER_ITER = 10
	BATCH_SIZE = 30
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-4
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 7 :    #contrastive loss
	SOFTMAX_METHOD = 14    #  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 1          #L-Softmax:2  A-Softmax:4 LMCL:2
	SOFTMAX_MARGIN = 2
	SHOW_PER_ITER = 250
	BATCH_SIZE = 64
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-3
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1*1e-1
