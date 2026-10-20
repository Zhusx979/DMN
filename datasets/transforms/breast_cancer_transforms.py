import torch
import math
import numbers
import random
import numpy as np
import pdb
import time
from skimage import measure
import cv2
import scipy

from torchvision import transforms

import albumentations as A


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.flip = A.HorizontalFlip(p=p)
        
    def __call__(self, sample):
        img = sample['image']
        img = self.flip(image=img)['image']
        sample['image'] = img
        return sample

class ElasticTransform(object):
    def __init__(self):
        self.method = A.ElasticTransform(alpha=1, sigma=5, alpha_affine=5)
    def __call__(self, sample):
        img = self.method(image=sample['image'])['image']
        sample['image'] = img
        return sample

class Sharpen(object):
    def __init__(self):
        self.method = A.Sharpen(lightness=(0.75, 1.0))
    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
            sample['image'] = img
        return sample

class RGBShift(object):
    def __init__(self):
        self.method = A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
        return sample

class GridDistortion(object):
    def __init__(self):
        self.method = A.GridDistortion()
    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
            sample['image'] = img
        return sample

class GaussNoise(object):
    def __init__(self):
        self.method = A.GaussNoise()
    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
            sample['image'] = img
        return sample

class FancyPCA(object):
    def __init__(self):
        self.method = A.FancyPCA()
    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
            sample['image'] = img
        return sample

class Blur(object):
    def __init__(self):
        self.method = A.Blur(blur_limit=(3,5))
    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
            sample['image'] = img
        return sample

class ClipMask(object):
    def __init__(self, size=128, flag=True):
        self.size = size
        self.flag = flag

    def __call__(self, sample):
        if not self.flag:
            return sample
        img = sample['image']
        image = np.array(img) 
        mask = np.zeros((image.shape[0], image.shape[1])) 
        boundingbox = sample['boundingbox']
        if type(boundingbox)!=np.ndarray:
            mask = np.load(boundingbox)
            if mask.shape[0] == 1:
                mask = mask.transpose(1,2,0)
            sample['boundingbox'] = np.array([0,0,0,0]).astype('int32')
        else:
            if(boundingbox[2]>boundingbox[0] and boundingbox[3]>boundingbox[1]):
                mask[boundingbox[1]:boundingbox[3], boundingbox[0]:boundingbox[2]] = np.mean(
                    image[boundingbox[1]:boundingbox[3], boundingbox[0]:boundingbox[2]], 2)
            mask =cv2.resize(mask, dsize=(self.size,self.size), interpolation=cv2.INTER_LINEAR) 
            mask = np.expand_dims(mask, 2)
        sample['mask0'] = mask
        return sample

class CropBlock(object):
    def __init__(self, size=32, block=8):
        self.size = size
        self.block = block

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        img_splits = []
        for i in range(8):
            for j in range(8):
                img_splits.append(img[i*32:(i+1)*32, j*32:(j+1)*32])
        sample['image'] = np.stack(img_splits, 0)
        sample['label'] = np.stack(8*8*[label], 0)
        return sample

class CropMask(object):
    def __init__(self, size=128, flag=True):
        self.size = size
        self.flag = flag

    def __call__(self, sample):
        if not self.flag:
            return sample
        img = sample['image']
        image = np.array(img) 
        boundingbox = sample['boundingbox']
        mask = np.zeros((image.shape[0], image.shape[1], 3)) 
        if(boundingbox[2]>boundingbox[0] and boundingbox[3]>boundingbox[1] and 
                boundingbox[2]<image.shape[0] and boundingbox[3]<image.shape[1]):
            midx = int((boundingbox[2]+boundingbox[0])/2)
            midy = int((boundingbox[3]+boundingbox[1])/2)
            bx = max(0, boundingbox[0])
            by = max(0, boundingbox[1])
            ex = min(boundingbox[2], image.shape[1])
            ey = min(boundingbox[3], image.shape[0])
            mask[by:ey, bx:ex] = 255 #image[by:ey, bx:ex]
        sample['mask0'] = mask.astype('uint8')
        return sample


class RandomCropMask(object):
    def __init__(self, final_size=256, crop_rate=1.5):
        self.final_size = final_size
        self.crop_rate = crop_rate

    def __call__(self, sample):
        img = sample['image']
        image = np.array(img)
        boundingbox = sample['boundingbox']
        self.crop_size = random.randint(self.final_size, int(self.crop_rate*self.final_size))
        mask = np.zeros((self.crop_size, self.crop_size, 3)) 
        # 如果图像尺寸太小，就同比例放大图像
        if image.shape[0] < self.crop_size:
            image = cv2.resize(image, dsize=(int(image.shape[1]/image.shape[0]*self.crop_size), self.crop_size), interpolation=cv2.INTER_LINEAR)
            boundingbox = (boundingbox /image.shape[0]*self.crop_size ).astype("int64")
        if image.shape[1] < self.crop_size:
            image = cv2.resize(image, dsize=(self.crop_size, int(image.shape[0]/image.shape[1]*self.crop_size)), interpolation=cv2.INTER_LINEAR)
            boundingbox = (boundingbox /image.shape[1]*self.crop_size).astype("int64")
        
        #print(boundingbox)
        if(boundingbox[2]>boundingbox[0] and boundingbox[3]>boundingbox[1] and 
                boundingbox[2]<image.shape[1] and boundingbox[3]<image.shape[0]):
            midx = int((boundingbox[2]+boundingbox[0])/2)
            midy = int((boundingbox[3]+boundingbox[1])/2)
            bx = max(0, int(boundingbox[2]-self.crop_size))
            by = max(0, int(boundingbox[3]-self.crop_size))
            ex = min(boundingbox[0], image.shape[1]-self.crop_size)
            ey = min(boundingbox[1], image.shape[0]-self.crop_size)
            if ex < bx:
                rx = max(0, min(midx - int(self.crop_size/2), image.shape[1]-self.crop_size))
            else:
                rx = random.randint(bx, ex)
            if ey < by:
                ry = max(0, min(midy - int(self.crop_size/2), image.shape[0]-self.crop_size))
            else:
                ry = random.randint(by, ey)
            final_img = image[ry:ry+self.crop_size, rx:rx+self.crop_size]
            #print(final_img.shape, boundingbox)
            mask[boundingbox[1]-ry:boundingbox[3]-ry, boundingbox[0]-rx:boundingbox[2]-rx] = 255
        else:
            #print("find a no bounding box image, its label is : ", sample["label"])
            # 如果框条件不满足就随机选一个框，此时标签应为1类
            rx = random.randint(0, image.shape[1]-self.crop_size)
            ry = random.randint(0, image.shape[0]-self.crop_size)
            final_img = image[ry:ry+self.crop_size,rx:rx+self.crop_size]
        
        sample['image'] = final_img.astype("uint8")
        sample['mask0'] = mask.astype("uint8")
        # if final_img.shape[0] != 256 or final_img.shape[1] != 256:
        #     print(rx, ry, final_img.shape, image.shape)
        return sample


class RandomMultiScaleCrop(object):
    def __init__(self, final_size=256, crop_rates=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]):
        self.final_size = final_size
        self.crop_rates = crop_rates

    def __call__(self, sample):
        img = sample['image']
        image = np.array(img)
        boundingbox = sample['boundingbox']

        images = []
        masks = []
        for crop_rate in self.crop_rates:
            self.crop_size = int(crop_rate*self.final_size)
            mask = np.zeros((self.crop_size, self.crop_size, 1)).copy()
            # 如果图像尺寸太小，就同比例放大图像
            if image.shape[0] < self.crop_size:
                image = cv2.resize(image, dsize=(int(image.shape[1]/image.shape[0]*self.crop_size), self.crop_size), interpolation=cv2.INTER_LINEAR)
                boundingbox = boundingbox /image.shape[0]*self.crop_size 
            if image.shape[1] < self.crop_size:
                image = cv2.resize(image, dsize=(self.crop_size, int(image.shape[0]/image.shape[1]*self.crop_size)), interpolation=cv2.INTER_LINEAR)
                boundingbox = boundingbox /image.shape[1]*self.crop_size
            
            if(boundingbox[2]>boundingbox[0] and boundingbox[3]>boundingbox[1] and 
                    boundingbox[2]<image.shape[1] and boundingbox[3]<image.shape[0]):
                midx = int((boundingbox[2]+boundingbox[0])/2)
                midy = int((boundingbox[3]+boundingbox[1])/2)
                bx = max(0, int(boundingbox[2]-self.crop_size))
                by = max(0, int(boundingbox[3]-self.crop_size))
                ex = min(boundingbox[0], image.shape[1]-self.crop_size)
                ey = min(boundingbox[1], image.shape[0]-self.crop_size)
                if ex < bx:
                    rx = max(0, min(midx - int(self.crop_size/2), image.shape[1]-self.crop_size))
                else:
                    rx = random.randint(bx, ex)
                if ey < by:
                    ry = max(0, min(midy - int(self.crop_size/2), image.shape[0]-self.crop_size))
                else:
                    ry = random.randint(by, ey)
                final_img = image[ry:ry+self.crop_size, rx:rx+self.crop_size]
                mask[boundingbox[1]-ry:boundingbox[3]-ry, boundingbox[0]-rx:boundingbox[2]-rx] = 255
            else:
                #print("find a no bounding box image, its label is : ", sample["label"])
                # 如果框条件不满足就随机选一个框，此时标签应为1类
                rx = random.randint(0, image.shape[1]-self.crop_size)
                ry = random.randint(0, image.shape[0]-self.crop_size)
                final_img = image[ry:ry+self.crop_size,rx:rx+self.crop_size]

            cur_img = cv2.resize(final_img, dsize=(self.final_size, self.final_size), interpolation=cv2.INTER_LINEAR) 
            images.append(np.mean(cur_img, 2))
            cur_mask = cv2.resize(mask, dsize=(self.final_size, self.final_size), interpolation=cv2.INTER_NEAREST) 
            masks.append(cur_mask)

        
        sample['image'] = np.stack(images, 2).astype("uint8")
        sample['mask0'] = np.stack(masks, 2).astype("uint8")
        # if final_img.shape[0] != 256 or final_img.shape[1] != 256:
        #     print(rx, ry, final_img.shape, image.shape)
        return sample


class CenterCropMask(object):
    def __init__(self, final_size=320):
        self.final_size = final_size

    def __call__(self, sample):
        img = sample['image']
        image = np.array(img)  
        boundingbox = sample['boundingbox']

        if image.shape[0] < self.final_size:
            image = cv2.resize(image, dsize=(int(image.shape[1]/image.shape[0]*self.final_size), self.final_size), interpolation=cv2.INTER_LINEAR)
            boundingbox = boundingbox /image.shape[0]*self.final_size 
        if image.shape[1] < self.final_size:
            image = cv2.resize(image, dsize=(self.final_size, int(image.shape[0]/image.shape[1]*self.final_size)), interpolation=cv2.INTER_LINEAR)
            boundingbox = boundingbox /image.shape[1]*self.final_size

        mask = np.zeros((self.final_size, self.final_size, 3)) 
        if(boundingbox[2]>boundingbox[0] and boundingbox[3]>boundingbox[1] and 
                boundingbox[2]<image.shape[1] and boundingbox[3]<image.shape[0]):

            midx = int((boundingbox[2]+boundingbox[0])/2)
            midy = int((boundingbox[3]+boundingbox[1])/2)
            bx = max(0, int(midx-self.final_size/2))
            by = max(0, int(midy-self.final_size/2))
            ex = min(boundingbox[0], image.shape[1]-self.final_size)
            ey = min(boundingbox[1], image.shape[0]-self.final_size)
            if ex < bx:
                rx = max(0, min(midx - int(self.final_size/2), image.shape[1]-self.final_size))
            else:
                rx = bx
            if ey < by:
                ry = max(0, min(midy - int(self.final_size/2), image.shape[0]-self.final_size))
            else:
                ry = by
            final_img = image[ry:ry+self.final_size, rx:rx+self.final_size]
            mask[boundingbox[1]-ry:boundingbox[3]-ry, boundingbox[0]-rx:boundingbox[2]-rx] = 255
        else:
            #print("find a no bounding box image, its label is : ", sample["label"])
            fx = random.randint(0, image.shape[1]-self.final_size)
            fy = random.randint(0, image.shape[0]-self.final_size)
            final_img = image[fx:fx+self.final_size,fy:fy+self.final_size]
        
        sample['image'] = final_img.astype("uint8")
        sample['mask0'] = mask.astype("uint8")
        return sample


class ClipMargin(object):
    def __init__(self, min_value=200, min_dis=5, split_dis=3, size=160):
        self.min_dis = min_dis
        self.split_dis = split_dis
        self.min_value = min_value
        self.size = size
        
    def __call__(self, sample):
        img = sample['image']
        image = np.array(img) 
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (3, 3)) 
        (_, thresh) = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        w, h = closed.shape[0], closed.shape[1]
        w_left = self.min_dis
        w_right = w-self.min_dis
        w_last_value = 0
        for idx in range(w//2, self.min_dis, -self.split_dis):
            #print(idx, np.sum(closed[idx:idx+self.split_dis]>0))
            if idx == w//2:
                w_last_value = np.sum(closed[idx:idx+self.split_dis] > 0)
                continue
            if np.sum(closed[idx:idx+self.split_dis] > 0) < w_last_value - self.min_value:
                w_left = idx
                break
            w_last_value = np.sum(closed[idx:idx+self.split_dis] > 0)
            
        for idx in range(w//2, w-self.min_dis, self.split_dis):
            if idx == w//2:
                w_last_value = np.sum(closed[idx:idx+self.split_dis] > 0)
                continue
            if np.sum(closed[idx:idx+self.split_dis] > 0) < w_last_value - self.min_value:
                w_right = idx
                break
            w_last_value = np.sum(closed[idx:idx+self.split_dis] > 0)
            
        h_left = self.min_dis
        h_right = h-self.min_dis
        h_last_value = 0
        for idx in range(h//2, self.min_dis, -self.split_dis):
            if idx == h//2:
                h_last_value = np.sum(closed[:, idx:idx+self.split_dis] > 0)
                continue
            if np.sum(closed[:,idx:idx+self.split_dis] > 0) < h_last_value - self.min_value:
                h_left = idx
                break
            h_last_value = np.sum(closed[:,idx:idx+self.split_dis] > 0)
            
        for idx in range(h//2, h-self.min_dis, self.split_dis):
            if idx == h//2:
                h_last_value = np.sum(closed[:,idx:idx+self.split_dis] > 0)
                continue
            if np.sum(closed[:,idx:idx+self.split_dis] > 0) < h_last_value - self.min_value:
                h_right = idx
                break
            h_last_value = np.sum(closed[:,idx:idx+self.split_dis] > 0)
        if h_right < h-150:
            h_right=h-150
    
        cimg = image[w_left:w_right,h_left:h_right]
        if cimg.shape[0]>self.size and cimg.shape[1]>self.size:        
            sample['image'] = cimg
        else:
            sample['image'] = image
        return sample
    
class RandomAffine(object):
    def __init__(self, degrees=5, onlyT=False):
        self.aff = transforms.RandomAffine(degrees)
        self.onlyT = onlyT
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if (self.onlyT and label == 1) or (not self.onlyT):
            img = self.aff(img)

        sample['image'] = img
        return sample
    
class RandomCrop(object):
    def __init__(self, rate=0.15):
        self.rate = rate
    def __call__(self, sample):
        img = sample['image']
        mask0 = sample['mask0']
        boundingbox = sample['boundingbox']
        
        center_rate = random.random()*0.08
        bx = random.randint(0,min(int(img.shape[1]*self.rate), boundingbox[0]))
        bx = max(int(img.shape[1]*center_rate), bx)
        ex = random.randint(max(img.shape[1]-int(img.shape[1]*self.rate), boundingbox[2]), img.shape[1])
        ex = min(img.shape[1]-int(img.shape[1]*center_rate), ex)
        by = random.randint(0,min(int(img.shape[0]*self.rate), boundingbox[1]))
        by = max(int(img.shape[0]*center_rate), by)
        ey = random.randint(max(img.shape[0]-int(img.shape[0]*self.rate), boundingbox[3]), img.shape[1])
        ey = min(img.shape[0]-int(img.shape[0]*center_rate), ey)
        
        sample['image'] = img[by:ey, bx:ex]
        sample['mask0'] = mask0[by:ey, bx:ex]
        return sample

class CenterCrop(object):
    def __init__(self, rate=0.1):
        self.rate = rate
    def __call__(self, sample):
        img = sample['image']
        mask0 = sample['mask0']
        boundingbox = sample['boundingbox']
        
        center_rate = 0.1
        bx = int(img.shape[1]*center_rate)
        ex = img.shape[1]-int(img.shape[1]*center_rate)
        by = int(img.shape[0]*center_rate)
        ey = img.shape[0]-int(img.shape[0]*center_rate)
        
        sample['image'] = img[by:ey, bx:ex]
        sample['mask0'] = mask0[by:ey, bx:ex]
        return sample

class Resize(object):
    def __init__(self, max_size=224):
        self.resize = A.LongestMaxSize(max_size=max_size)
        self.pad = A.PadIfNeeded(min_height=max_size, min_width=max_size)
        
    def __call__(self, sample):
        image = self.resize(image=sample['image'])['image'] # albumentations的resize返回值为dict
        image = self.pad(image = image)['image']
        sample['image'] = image
        return sample
    
class ColorAug(object):
    """
    augment the color contrast, brightness, and saturation
    """
    def __init__(self):
        self.method = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    def __call__(self, sample):
        if('image' in sample.keys()):
            img = sample['image']
            img = self.method(image=img)['image']
            sample['image'] = img
        return sample
    
    
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        img = np.array(sample['image']).astype('float32')
        img /= 255.0
        sample['image'] = img
        return sample


class ToTensor(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        img = np.array(sample['image']).astype('float32') 
        sample['image'] = img.transpose(2,0,1)
        return sample