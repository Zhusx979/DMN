import torch
import math
import numbers
import random, pdb
import numpy as np
from PIL import Image, ImageOps
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return img
        
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        
        return img

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))

        return img

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class Normalize_divide(object):
    def __init__(self, denominator = 255.0):
        self.denominator = float(denominator)
    
    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        
        return img

class Normalize(object):
    """使用预定义的全局均值和标准差进行标准化，而不是每图像独立计算"""
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        # 如果图像是单通道，将 mean/std 转换为单值
        if img.ndim == 2:
            # self.mean/std are np arrays (see __init__) -> take channel 0 as scalar
            mean_val = float(self.mean[0])
            std_val = float(self.std[0])
            img = (img - mean_val) / std_val
        else:
            # 确保通道维在最后
            img = (img - self.mean) / self.std
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass
    
    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        if len(img.shape) == 3:
            img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        elif len(img.shape) == 2:
            img = np.expand_dims(np.array(img).astype(np.float32), -1).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img):
        img = img.resize(self.size, Image.BILINEAR)
        
        return img

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                
                return img

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(img))
        return img

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)

        return img

class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return img
        
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        
        return img

class Padding(object):
    """padding zero to image to match the maximum value of target width or height"""
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
    
    def __call__(self, img):
        w, h = img.size
        target_h, target_w = self.size
        
        left = top = right = bottom = 0
        doit = False
        if target_w > w:
            delta = target_w - w
            left = delta // 2
            right = delta - left
            doit = True
            
        if target_h > h:
            delta = target_h - h
            top = delta // 2
            bottom = delta - top
            doit = True
        if doit:
            img = ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)
            
        return img

class RandomSized(object):
    def __init__(self, size, scale_min, scale_max):
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.padding = Padding(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img):
        
        scale = random.uniform(self.scale_min, self.scale_max)
        
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])
        
        img = img.resize((w, h), Image.BILINEAR)
        
        padded = self.padding(img)
        cropped = self.crop(padded)
        return cropped

class RandomScale(object):
    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, img):
        
        scale = random.uniform(self.scale_min, self.scale_max)
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])
        
        img = img.resize((w, h), Image.BILINEAR)

        return img
