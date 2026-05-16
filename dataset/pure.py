import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CDData(object):
    def __init__(self, meta_file="", data_root=''):
        lines = open(meta_file,'r').readlines()
        imgs1, imgs2, labels = [], [], []
        for line in lines:
            s = line.strip().split('\t')
            ss = [os.path.join(data_root, _s) for _s in s]
            imgs1.append(ss[0])
            imgs2.append(ss[1])
            labels.append(ss[2])
        self.__dict__.update(locals())

    def get_img1(self, idx):
        img = cv2.imread(self.imgs1[idx], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_img2(self, idx):
        img = cv2.imread(self.imgs2[idx], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_label(self, idx):
        img = cv2.imread(self.labels[idx], -1)
        return img

    def get_img12(self, idx):
        img1 = cv2.imread(self.imgs1[idx], -1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.imgs2[idx], -1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        return img1,img2

    def show_img12(self, idx):
        img1, img2 = self.get_img12(idx)
        plt.subplot(1,2,1);plt.imshow(img1)
        plt.subplot(1,2,2);plt.imshow(img2)
        plt.show()
        return

    def show_all(self, idx):
        img1, img2 = self.get_img12(idx)
        label = self.get_label(idx)
        plt.subplot(1,3,1);plt.imshow(img1)
        plt.subplot(1,3,2);plt.imshow(img2)
        plt.subplot(1,3,3);plt.imshow(label,cmap='gray')
        plt.show()
        return
    

class CDFData(object):
    def __init__(self, meta_file="", data_root=''):
        lines = open(meta_file,'r').readlines()
        imgs1, imgs2, labels, feats1, feats2 = [], [], [], [], []
        for line in lines:
            s = line.strip().split('\t')
            ss = [os.path.join(data_root, _s) \
                if not _s.startswith('/') else _s for _s in s]
            imgs1.append(ss[0])
            imgs2.append(ss[1])
            labels.append(ss[2])
            feats1.append(ss[3])
            feats2.append(ss[4])
        self.__dict__.update(locals())

    def get_feat1(self, idx):
        feat = np.load(self.feats1[idx], allow_pickle=True).tolist()
        return feat

    def get_feat2(self, idx):
        feat = np.load(self.feats2[idx], allow_pickle=True).tolist()
        return feat
    
    def get_feat12(self, idx):
        feat1 = np.load(self.feats1[idx], allow_pickle=True).tolist()
        feat2 = np.load(self.feats2[idx], allow_pickle=True).tolist()
        return feat1, feat2

    def get_img1(self, idx):
        img = cv2.imread(self.imgs1[idx], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_img2(self, idx):
        img = cv2.imread(self.imgs2[idx], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_label(self, idx):
        img = cv2.imread(self.labels[idx], -1)
        return img

    def get_img12(self, idx):
        img1 = cv2.imread(self.imgs1[idx], -1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.imgs2[idx], -1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        return img1,img2
    
    def get_all(self, idx):
        img1 = cv2.imread(self.imgs1[idx], -1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.imgs2[idx], -1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.labels[idx], -1)
        feat1 = np.load(self.feats1[idx], allow_pickle=True).tolist()
        feat2 = np.load(self.feats2[idx], allow_pickle=True).tolist()
        return img1,img2,label,feat1,feat2

    def show_img12(self, idx):
        img1, img2 = self.get_img12(idx)
        plt.subplot(1,2,1);plt.imshow(img1)
        plt.subplot(1,2,2);plt.imshow(img2)
        plt.show()
        return

    def show_all(self, idx):
        img1, img2 = self.get_img12(idx)
        label = self.get_label(idx)
        plt.subplot(1,3,1);plt.imshow(img1)
        plt.subplot(1,3,2);plt.imshow(img2)
        plt.subplot(1,3,3);plt.imshow(label,cmap='gray')
        plt.show()
        return
    