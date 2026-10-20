
import os
import numpy as np
from torch.utils.data import Dataset
import pdb
import time
import cv2
import scipy
import pandas as pd
from shutil import copyfile
from loguru import logger
import random

class BCData(Dataset):
    def __init__(self, data_dict, class_ratio, transform=None):
        self.transform = transform
        self.class_idxs, self.image_paths = [], []
        for (class_idx, txt_paths) in data_dict.items():
            image_paths = []
            for txt_path in txt_paths:
                image_paths += self._read_txt(txt_path)
            
            num_ratio = class_ratio[class_idx]
            # 过采样后的样本数量
            image_num = int(len(image_paths) * num_ratio)
            
            for i in range(image_num):
                self.class_idxs.append(class_idx)
                image_path = image_paths[i%len(image_paths)]
                self.image_paths.append(image_path)
            
            logger.info("Class: {}, number of images: {}".format(class_idx, image_num))
        
        idxs_paths = list(zip(self.class_idxs, self.image_paths))
        random.shuffle(idxs_paths)
        self.class_idxs, self.image_paths = zip(*idxs_paths)
        
        
        # idxs = list(range(len(self.image_paths)))
        # random.shuffle(idxs)
        # self.class_idxs = self.class_idxs[idxs]
        # self.image_paths = self.image_paths[idxs]
            
    
    def _read_txt(self, txt_path):
        fp = open(txt_path, "r")
        lines = fp.readlines()
        fp.close()
        image_paths = []
        for line in lines:
            if line.strip()!="":
                image_paths.append(line.strip())
        return image_paths
    
    
    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.class_idxs[idx]
        
        sample = {'image': image, 'label': label}
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample