import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TBX11KDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', split_ratio=(0.8, 0.1, 0.1), seed=911, balance_count=800):
        """
        Args:
            root_dir: Path to the 'imgs' directory containing 'sick', 'health', 'tb'.
            transform: Transformations to apply.
            mode: 'train', 'val', or 'test'.
            split_ratio: (train, val, test) ratios.
            seed: Random seed for reproducibility.
            balance_count: Number of images to select from each class.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['health', 'sick', 'tb']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        random.seed(seed)
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Directory {cls_dir} does not exist.")
                continue
                
            all_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_imgs.sort() # Ensure reproducibility before sampling
            
            # Balance to balance_count (800)
            if len(all_imgs) > balance_count:
                selected_imgs = random.sample(all_imgs, balance_count)
            else:
                selected_imgs = all_imgs
            
            # Shuffle selected images before splitting
            random.shuffle(selected_imgs)
            
            # Split
            num_imgs = len(selected_imgs)
            train_end = int(num_imgs * split_ratio[0])
            val_end = train_end + int(num_imgs * split_ratio[1])
            
            if mode == 'train':
                imgs = selected_imgs[:train_end]
            elif mode == 'val':
                imgs = selected_imgs[train_end:val_end]
            else: # test
                imgs = selected_imgs[val_end:]
            
            for img in imgs:
                self.samples.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_name = os.path.basename(path)
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return [img, label, img_name]


if __name__ == '__main__':
    tbx11k_root = "E:/python/Project/Dataset/TBX11K/imgs"
    trainset = TBX11KDataset(tbx11k_root, mode='train')
    valset = TBX11KDataset(tbx11k_root, mode='val')
    testset = TBX11KDataset(tbx11k_root,  mode='test')
    print(f"len of training: {len(trainset)}, len of validation: {len(valset)}, len of test: {len(testset)}")