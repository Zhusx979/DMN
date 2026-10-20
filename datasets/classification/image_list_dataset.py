import os, pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

class Image_dataset(Dataset):
    def __init__(self, 
                 data_tuples,
                 transform = None):
        """The dataset for single image.
        Args:
            data_tuples: list of [label:int, data_root:str, list_txt_path:str]
        """
        self.transform = transform
        self.img_paths, self.labels = [], []
        for [label, data_root, list_txt] in data_tuples:
            fp = open(list_txt, "r")
            paths = fp.readlines()
            fp.close()
            paths = [os.path.join(data_root, img_path.strip()) for img_path in paths if img_path.strip() != "" and not img_path.startswith("#")]
            
            self.img_paths += paths
            self.labels += [label]*len(paths)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)
        label = self.labels[idx]
        # zk_OCT / zk_pneumonia jpegs are grayscale ('L'); the paper protocol
        # feeds 3-channel inputs, so replicate channels (no-op for RGB files).
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        return [img, label, img_name]