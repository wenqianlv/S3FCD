import os
import random

import cv2
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from .operations import AugOps


class SYSUDataset(Dataset):
    """SYSC change detection dataset.
    """
    def __init__(self, 
                 metafile="", 
                 data_root="",
                 pipeline=None,
                 test_mode=False,
                 c255t1_in_mask=False,
                 ):
        super().__init__()
        lines = open(metafile).readlines()
        images = []
        feats = []
        for line in lines:
            item = line.strip().split('\t')
            s = item[0]
            if not s.startswith('/'):
                s = os.path.join(data_root, s)
            images.append(s)
            
            if len(item) > 1:
                f = item[1]
                if not f.startswith('/'):
                    f = os.path.join(data_root, f)
                feats.append(f)
        # import pdb;pdb.set_trace()
        self.images = images
        self.feats = feats
        self.augops = AugOps()
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.c255t1_in_mask = c255t1_in_mask
        self.tensor = torchvision.transforms.ToTensor()
        # self.__dict__.update(locals())

    def __getitem__(self, index):
        img_path = self.images[index]
        img = cv2.imread(img_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # get feat
        if self.feats:
            feat_path = self.feats[index]
            feat = np.load(feat_path, allow_pickle=True).tolist()
        else:
            feat = None
        # generate image1
        img1 = img.copy()
        # generate image2 and mask
        img2_pre = img.copy()
        p = random.random()
        if p < 0.5:
            img2_pre = self.augops.hist_brightness_aug(img2_pre, minv=0.5,
                                                        maxv=1.5, hwidth=16)

        
        p = random.random()
        if p < 0.5:
            img2 = img2_pre
            mask = 1-(img2_pre == img2_pre).astype(np.float32)
            
        elif p < 0.99:  # use shuffle
            bboxes = self.augops.get_bboxes(nums=20, img_hw=(h, w),
                                            region_hw=(120, 120))
            img2_pre, mask = self.augops.shuffle_trans(img=img2_pre,
                                                     bboxes=bboxes,
                                                     thr=0.005,
                                                     win_size=11,
                                                     img_ref=img,
                                                     feat=feat,
                                                     )


        else:  # use jigsaw
            img2_jigsaw = self.augops.jigsaw_trans(img2_pre, nr=(2, 5),
                                                   nc=(2, 5))
            mask = 1-(img2_pre == img2_jigsaw).astype(np.float32)
            img2_pre = img2_jigsaw.copy()
            
        p = random.random()
        if p < 0.5:
            img2_pre = self.augops.shift_image(img2_pre, 
                                               dx=random.randint(0,10),
                                               dy=random.randint(0,10),
                                               )
        
        img2 = img2_pre
        if self.pipeline == None:
            return img1, img2, mask
        
        # img2 = self.augops.style_transfer(img2, img1)
        
        augmented = self.pipeline(image=img1, image1=img2, mask=mask)
        image0 = self.tensor(augmented['image'])
        image1 = self.tensor(augmented['image1'])
        if self.test_mode:
            return image0, image1
        else:
            mask = self.tensor(augmented['mask'])[0].long()
            return image0, image1, mask

    def generate_mask_from_bboxes(self, img, bboxes):
        img_new = np.zeros_like(img, dtype=np.float32)
        for (x, y, dx, dy) in bboxes:
            img_new[x:x+dx, y:y +
                    dy] = 1.0
        return img_new

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    data_root = "/Users/shinian/proj/data/SYSU-CD/"
    metafile = "/Users/shinian/proj/data/SYSU-CD/val.txt"
    dataset = SYSUDataset(metafile=metafile, data_root=data_root)
    idx = random.randint(0,len(dataset))
    img1, img2, mask = dataset[idx]
    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1);plt.imshow(img1)
    plt.subplot(1,3,2);plt.imshow(img2)
    plt.subplot(1,3,3);plt.imshow(mask,cmap='gray')
    plt.show()



