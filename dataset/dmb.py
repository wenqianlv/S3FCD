import random
from collections import deque

import cv2
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from einops import rearrange


from utils import is_gray


class DataMemoryBank(object):
    
    def __init__(self, cap=None, fe=None) -> None:
        super().__init__()
        self.cap = cap
        self.fe = fe
        self.mem = deque([], maxlen=self.cap) # [[img, emb1(ssim), emb2(feature)]]
    
    def clear(self):
        del self.mem
        self.mem = deque([], maxlen=self.cap)
    
    def push(self, samples, feats=None): # samples [patch1, patch2]
        # print(feats)
        if not isinstance(samples, list):
            samples = [samples]
        emb1s = self.get_emb(samples, choice='emb1')
        emb2s = self.get_emb(samples, choice='emb2') if feats==None or not feats else feats
        items = [list(item) for item in zip(samples, emb1s, emb2s)]
        self.mem.extend(items)
        return items
        
    def sample_from_memory(self, n=None):
        if n is None:
            samples = self.mem.copy()
        elif n <= len(self.mem):
            samples = random.sample(self.mem, n)
        else:
            logger.warning(f'try sample ({n}) from ({len(self.mem)}). return all')
            samples = self.mem.copy()
        return samples
    
    def get_imgs(self):
        imgs = [i[0] for i in self.mem]
        return imgs
    
    def get_emb(self, samples, choice='emb1'):
        if not isinstance(samples, list):
            samples = [samples]
            
        if choice in ['emb1']: 
            embs = [sample if is_gray(sample) \
                else cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)  for sample in samples]
        elif choice in ['emb2']:
            embs = samples
        else:
            raise NotImplementedError(f"wrong: {self.fe}/{choice}")
        return embs
    
    def update_emb2(self): # update all emb2s
        if self.fe is None:
            return
        emb2s = self.get_emb(self.get_imgs(), choice='emb2')
        for idx in range(len(self.mem)):
            self.mem[idx][2] = emb2s[idx]
            
    def update_fe(self, fe):
        self.fe = fe
    
    
    def compare_emb1(self, query, keys, method='ssim'):
        if method == 'ssim':
            h,w = query.shape[:2]
            keys = [cv2.resize(key, (w,h)) for key in keys]
            
            dists = [structural_similarity(query,key,win_size=7) for key in keys]
        return dists
    
    
    def compare_emb2(self, query, keys, method='cos'):
        # if method == 'cos':
        #     if isinstance(keys,list): keys = np.stack(keys, axis=0)
        #     dists = cosine_similarity(query, keys)
        # when query is feat and keys are [feats]
        if method == 'cos':
            dists = []
            for key in keys:
                dist = 0
                for fk,fq in zip(key[:], query[:]):
                    # print(fk[0].shape,fq[0].shape,len(fk),len(fq))
                    # print(fk.shape, fq.shape,'fq,fk')
                    c,h,w = fq.shape
                    fq = rearrange(fq, 'c h w -> (h w) c')
                    fk = rearrange(fk, 'c h w -> (h w) c')
                    if fq.shape != fk.shape:
                        fq = fq.mean(0)
                        fk = fk.mean(0)
                    dist += cosine_similarity(fq.reshape(1, -1), fk.reshape(1, -1)).sum()
                    
                    # for i in range(h*w):
                    #     dist += cosine_similarity(fq[i].reshape(1, -1), fk[i].reshape(1, -1)).sum()
                dists.append(dist)
        
        
        return dists

    
    def get_pos_neg(self, img, npos=2, nneg=2, base=1000, use='emb1', return_img=True, push_img=True, feat=None):
        samples = self.sample_from_memory(n=None)
        if use == 'emb1':
            query = self.get_emb([img], choice='emb1')[0]
            emb1s = [item[1] for item in samples]
            similars = self.compare_emb1(query, emb1s, method='ssim')

        elif use == 'emb2':
            # import pdb;pdb.set_trace()
            samples = [item for item in samples if item[0].shape==img.shape]
            random.shuffle(samples)
            samples = samples[:base]
            query = self.get_emb([feat if feat!=None else img], choice='emb2')[0]
            emb2s = [item[2] for item in samples]
            # print(emb2s)
            # filter by the shape
            similars = self.compare_emb2(query, emb2s, method='cos')
        
        # print(similars)
        argsorts = np.argsort(similars).tolist() # small --> large
        # print(argsorts)
        if len(argsorts) < npos or len(argsorts) < nneg:
            logger.warning(f'try sample ({npos}/{nneg}) from ({argsorts}) in len(samples)-({len(samples)}). return 0')
            pos_idx = 0
            neg_idx = 0
            return None, None
        else:
            pos_idx = random.sample(argsorts[-npos:],1)[0]
            neg_idx = random.sample(argsorts[:nneg],1)[0]
        pos = samples[pos_idx]
        neg = samples[neg_idx]
        
        if push_img:
            self.push([img],feats=[feat])
        
        if return_img:
            pos = pos[0]
            neg = neg[0]
            h,w = img.shape[:2]
            pos = cv2.resize(pos, (w,h))
            neg = cv2.resize(neg, (w,h))
            
        return pos, neg
        