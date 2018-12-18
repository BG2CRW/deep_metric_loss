# deep metric loss in PyTorch 1.0

This project aims at providing some typically deep metric losses in recent years and we run them in COU200-2011,CARS196 and Stanford Online Product datasets using PyTorch 1.0.

## LOSS
- **smooth prec@k loss** 
- **angular loss** 
- **clustering loss(TODO)** 
- **npair loss** 
- **lifted loss** 
- **semi-hard triplet loss** 
- **contrastive loss** 
- **arcface loss** 
- **cosface loss(LMCL)** 
- **proxyNCA** 
- **A-softmax loss** 
- **L-softmax loss** 
- **softmax loss** 

## DATASETS
- **auto downloader and spliter** 
- **CUB200-2010** 
- **CUB200-2011(with or without bounding box)** 
- **CARS196(with or without bounding box)** 
- **Stanford Online Product** 

## VALIDATIONS
- **precsion@k** 
- **recall@k** 
- **NMI(TODO)** 
- **F1(TODO)** 

## TOOLS
- **TSNE** 

```bash
conda install pytorch torchvision -c pytorch
# install pytorch1.0
cd deep_metric_loss
# by default, it runs on the GPU
pip install -r requirements.txt

#choose right config file in main.py and set it whether you would download and split download and whether you need the dataset with bounding boxes in *_config.py
python main.py

#draw TSNE picture
#choose right config file in test_and_tsne.py and set it whether you would download and split download and whether you need the dataset with bounding boxes in *_config.py
```
