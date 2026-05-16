import os

import cv2
import torchvision
from torch.utils.data import Dataset

import numpy as np
def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:,:,i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:,:,i])
        target_image_fshift = np.fft.fftshift(target_image_f)
        
        change_length = 1
        source_image_fshift[int(h/2)-change_length:int(h/2)+change_length, 
                            int(h/2)-change_length:int(h/2)+change_length] = \
            target_image_fshift[int(h/2)-change_length:int(h/2)+change_length,
                                int(h/2)-change_length:int(h/2)+change_length]
            
        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        
        source_image_if[source_image_if>255] = np.max(source_image[:,:,i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1,0).swapaxes(1,2)
    out = out.astype(np.uint8)
    return out


class CommonDataset(Dataset):
    """A common dataset class for two input.
    metafile:
        image_000_a.jpg image_000_b.jpg gt_000.png xxx
        image_001_a.jpg image_001_b.jpg gt_001.png xxx
        image_002_a.jpg image_002_b.jpg gt_002.png xxx
        ...

    Args:
        metafile (str): Path to meta_file.
        pipeline (list[dict]): Processing pipeline.
        data_root (str): Data root for images.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        sep (str): Sep in metafile
        c255t1_in_mask (bool): Convert 255 in mask into 1
    """

    def __init__(self,
                 metafile,
                 data_root='',
                 pipeline=None,
                 test_mode=False,
                 sep='\t',
                 c255t1_in_mask=False,
                 st=False,
                 ):
        super().__init__()
        self.metafile = metafile
        self.data_root = data_root
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.sep = sep
        self.c255t1_in_mask = c255t1_in_mask
        self.data = self.get_data()
        self.tensor = torchvision.transforms.ToTensor()
        self.st = st

    def get_data(self):
        data = []
        lines = open(self.metafile).readlines()
        for line in lines:
            s = line.strip().split(' ')
            assert len(s) >= 2, f"wrong when processing {line}"
            s[:2] = [os.path.join(self.data_root, item) for item in s[:2]]
            if not self.test_mode:
                s[2] = os.path.join(self.data_root, s[2])
            data.append(s)
        return data

    def read_images(self, paths):
        images = [cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
                  for path in paths]
        if self.st:
            images[1] = style_transfer(images[1], images[0])
        return images

    def read_mask(self, path):
        mask = cv2.imread(path, -1)
        if self.c255t1_in_mask and mask.max() > 1:
            mask = mask / 255
        mask = mask.astype('int64')
        return mask

    def process(self, image0, image1, mask=None):
        # import pdb; pdb.set_trace()
        if self.pipeline == None:
            return image0, image1, mask
        if mask is None:
            augmented = self.pipeline(image=image0, image1=image1)
            image0 = self.tensor(augmented['image'])
            image1 = self.tensor(augmented['image1'])
            return image0, image1
        else:
            augmented = self.pipeline(image=image0, image1=image1, mask=mask)
            image0 = self.tensor(augmented['image'])
            image1 = self.tensor(augmented['image1'])
            mask = self.tensor(augmented['mask'])[0].long()
            return image0, image1, mask

    def get_file_name(self, idx, suffix=None):
        items = self.data[idx]
        file_path = os.path.basename(items[0])
        if suffix:
            file_path = file_path.split('.')[0] + suffix
        return file_path
    
    def get_single_item(self, idx):
        item = self.data[idx]
        return item
        
            

    def __getitem__(self, idx):
        items = self.data[idx]
        image0, image1 = self.read_images(items[:2])
        mask = None if self.test_mode else self.read_mask(items[2])
        return self.process(image0, image1, mask)

    def __len__(self):
        return len(self.data)
