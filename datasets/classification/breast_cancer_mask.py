
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


class BCData(Dataset):
    def __init__(self, data_root, data_txt, transform=None):
        data_files = []
        boundingbox = []
        labels = []
        
        label_dist = np.zeros((3,))
        
        count = 0
        with open(data_txt, "r") as f:
            xs = f.readlines()
            for x in xs:
                anchor, cur_label, cur_path = self.general_process(x)
                if cur_label is None:
                    continue
                labels.append(cur_label)
                label_dist[cur_label] += 1
                data_files.append(os.path.join(data_root, cur_path.replace(":", "\uf03a")))
                boundingbox.append(anchor)
                count += 1
        logger.info("Dazhou have ", count)                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels

        self.transform = transform
        logger.info('the data length is %d' % (len(self.data_files)))
        logger.info("class 1/2->%d" % (label_dist[0]))
        logger.info("class 0/3->%d" % (label_dist[1]))
        logger.info("class 4/5->%d" % (label_dist[2]))


    def __len__(self):
        L = len(self.data_files)
        return L
    
    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        label_arr = x[1].split('\n')[0].split('_T_')
        
        if int(label_arr[0]) == 0: # 如果是1类，没有检测框，剔除掉
            return None, None, None

        if int(label_arr[0]) <2:
            cur_label = 0 # 低风险
        elif int(label_arr[0]) > 2:
            cur_label = 2 # 高风险
        else:
            cur_label = 1 # 中风险
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])

        return anchor, cur_label, x[0]


    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def __str__(self):
        pass