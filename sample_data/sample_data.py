import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import random
import torch
import os
import sample_data.download_split_data as D

def default_loader(path):
	try:
		img = Image.open(path)
		return img.convert('RGB')
	except:
		print("Cannot read image: {}".format(path))
class ourData(Dataset):
    def __init__(self, img_path, txt_path, data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.split()[1]) for line in lines]
            self.img_label = [int(line.split(' ')[0])-1 for line in lines]
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label, img_name

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make_transform(sz_resize = 256, sz_crop = 227, mean = [128, 117, 104], 
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True, 
        intensity_scale = [[0, 1], [0, 255]]):
    return transforms.Compose([
        transforms.Compose([ # train: horizontal flip and random resized crop
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
        ]) if is_train else transforms.Compose([ # test: else center crop
            transforms.Resize(sz_resize),
            transforms.CenterCrop(sz_crop),
        ]),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
        transforms.Lambda(
            lambda x: x[[2, 1, 0], ...]
        ) if rgb_to_bgr else Identity()
    ])


class Preprocess():
	def __init__(self,root,use_cuda,test_batch_size,method,dataset_name,with_bounding_box,download):
		self.root = os.path.join(root, dataset_name)
		self.use_cuda=use_cuda
		self.test_batch_size=test_batch_size
		self.method=method
		self.size = 227

		if 'CARS196' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_CARS196(root)
			train_img_path = self.root
			test_img_path = self.root
			train_txt_path = os.path.join(self.root,'train.txt')
			test_txt_path = os.path.join(self.root,'test.txt')
			if with_bounding_box==True:
				train_txt_path = os.path.join(self.root,'bounding_train.txt')
				test_txt_path = os.path.join(self.root,'bounding_test.txt')
			self.classes = 98
		if 'CUB200' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_CUB200(root)
			train_img_path = os.path.join(self.root,'images')
			test_img_path = os.path.join(self.root,'images')
			train_txt_path=os.path.join(self.root,'lists','new_train.txt')
			test_txt_path=os.path.join(self.root,'lists','new_test.txt')
			self.classes = 100
		if 'CUB200-2011' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_CUB200_2011(root)
			train_img_path = os.path.join(self.root,'CUB_200_2011')
			test_img_path = os.path.join(self.root,'CUB_200_2011')
			train_txt_path = os.path.join(self.root,'CUB_200_2011','new_train.txt')
			test_txt_path = os.path.join(self.root,'CUB_200_2011','new_test.txt')
			if with_bounding_box==True:
				train_txt_path = os.path.join(self.root,'CUB_200_2011','new_bounding_train.txt')
				test_txt_path = os.path.join(self.root,'CUB_200_2011','new_bounding_test.txt')
			self.classes = 100
		if 'Stanford_Online_Products' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_Stanford_Online_Products(root)
			train_img_path = self.root
			test_img_path = self.root
			train_txt_path = os.path.join(self.root,'new_train.txt')
			test_txt_path = os.path.join(self.root,'new_test.txt')
			self.classes = 11318
		print(train_txt_path)
		self.dataset_train = ourData(img_path=train_img_path,
										txt_path=train_txt_path,
										data_transforms=make_transform()) 
		self.dataset_test = ourData(img_path=test_img_path,
										txt_path=test_txt_path,
										data_transforms=make_transform(is_train=False)) 

		self.index_pointer_train=np.zeros((self.classes,), dtype=np.int)
		self.index_train=np.zeros((self.classes,200), dtype=np.int)
		
		f_train=open(train_txt_path)
		i=0
		for line in f_train:
			self.index_train[int(line.split(' ')[0])-1][self.index_pointer_train[int(line.split(' ')[0])-1]]=i
			self.index_pointer_train[int(line.split(' ')[0])-1]+=1
			i+=1	
		f_train.close()
		print(" dataset: "+dataset_name," with bounding_box: ",with_bounding_box)
		print("Total images in training sets: ",np.sum(self.index_pointer_train))
		self.test_loader = data.DataLoader(dataset=self.dataset_test,batch_size=self.test_batch_size,shuffle=False,num_workers = 16,pin_memory = True)
		self.test_train_loader = data.DataLoader(dataset=self.dataset_train,batch_size=self.test_batch_size,shuffle=False,num_workers = 16,pin_memory = True)

	def test_loader(self):
		return self.test_loader
	def test_train_loader(self):
		return self.test_train_loader

	def next_train_batch(self,n_pos,N, batch_size):
		if self.method==0:#prec@k loss
			flag=0  
			batch_img=torch.zeros(0,3,self.size,self.size)
			K_arr=[]	
			real_y=torch.zeros((N+1)*batch_size)
			y_=torch.zeros(0,N)
			for i in range(batch_size):
				classid=random.randint(0,self.classes-1) #random choose a class 1-11318 training class#11317
				temp=self.index_train[classid][:self.index_pointer_train[classid]]
				if n_pos+1>len(temp):
					flag=0
				if n_pos+1<len(temp):
					flag=1
				K=np.min([n_pos+1,len(temp)])-1
				K_arr.append(K)

				temp_y_=torch.cat([torch.ones(K),torch.zeros(N-K)],0).reshape(1,N)
				y_=torch.cat([y_,temp_y_],0)
				for j in range (K+1):
					real_y[j+i*(N+1)]=classid
				for k in range(N-K):
					while True:
						neg_id=random.randint(0,self.classes-1)
						if neg_id!=classid:
							break
					real_y[k+K+1+i*(N+1)]=neg_id
			pointer=0
			if flag==0:
				for i in range(batch_size):
					for j in range(K_arr[i]+1):
						temp_index=self.index_train[int(real_y[pointer])][j]
						temp_img=self.dataset_train[temp_index][0].reshape(1,3,self.size,self.size)
						batch_img=torch.cat([batch_img,temp_img],0)
						pointer+=1
					for j in range(N-K_arr[i]):
						temp_index=self.index_train[int(real_y[pointer])][random.randint(0,self.index_pointer_train[int(real_y[pointer])]-1)]
						temp_img=self.dataset_train[temp_index][0].reshape(1,3,self.size,self.size)
						batch_img=torch.cat([batch_img,temp_img],0)
						pointer+=1
			else:
				for i in range((N+1)*batch_size):
					batch_img=torch.cat([batch_img,self.dataset_train[self.index_train[int(real_y[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)],0)

		if self.method==5 or self.method==2:#lifted or clustering

			y_=torch.zeros(2*batch_size)
			for i in range(batch_size):
				y_[i]=random.randint(0,self.classes-1)
				y_[i+batch_size]=y_[i]
			real_y=y_
			for i in range(2*batch_size):
				if i==0:
					batch_img=self.dataset_train[self.index_train[int(y_[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)
				else:
					batch_img=torch.cat([batch_img,self.dataset_train[self.index_train[int(y_[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)],0)

		if self.method==7:#contrastive
			y_=torch.gt(torch.rand(batch_size),0.5)
			real_y=torch.zeros(2*batch_size)
			for i in range(batch_size):
				real_y[i]=random.randint(0,self.classes-1)
				if y_[i]==1:
					real_y[i+batch_size]=real_y[i]
				else:
					while True:
						real_y[i+batch_size]=random.randint(0,self.classes-1)
						if real_y[i+batch_size]!=real_y[i]:
							break
			for i in range(2*batch_size):
				if i==0:
					batch_img=self.dataset_train[self.index_train[int(real_y[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)
				else:
					batch_img=torch.cat([batch_img,self.dataset_train[self.index_train[int(real_y[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)],0)
		
		if self.method==3 or self.method==4 or self.method==6:#npair or angular
			y_=torch.from_numpy(np.array(random.sample(range(self.classes-1), batch_size)))
			y_=torch.cat([y_,y_],0)
			real_y=y_
			for i in range(2*batch_size):
				if i==0:
					batch_img=self.dataset_train[self.index_train[int(y_[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)
				else:
					batch_img=torch.cat([batch_img,self.dataset_train[self.index_train[int(y_[i])][random.randint(0,self.index_pointer_train[int(real_y[i])]-1)]][0].reshape(1,3,self.size,self.size)],0)
		
		return batch_img,y_,real_y.long()
