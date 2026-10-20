
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
from PIL import Image

class BCData(Dataset):
    def __init__(self, data_root, data_txt, transform=None):
        self.transform = transform
        self.class_idxs, self.image_paths = [], []
        self.image_paths, self.label_idxs = self._read_txt(data_root, data_txt)

        logger.info("Number of images: {}".format(len(self.image_paths)))
        
    
    def _read_txt(self, data_root, txt_path):
        fp = open(txt_path, "r")
        lines = fp.readlines()
        fp.close()
        image_paths, labels = [], []
        for line in lines:
            if line.strip()!="":
                image_name, label_idx = line.strip().split("\t")
                image_paths.append(os.path.join(data_root, image_name.replace(":", "\uf03a")))
                labels.append(int(label_idx))
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.label_idxs[idx]
        
        sample = {'image': image, 'label': label}
        
        if self.transform is not None:
            image = self.transform(image)
            
        return [image, label, image_path]