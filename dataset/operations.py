import cv2
import math
import random
import numpy as np
import albumentations as A

from imgaug import augmenters as iaa
from scipy.ndimage import rotate
from skimage.metrics import structural_similarity
from torch.multiprocessing  import Pool

from .dmb import DataMemoryBank



class AugOps(object):
    CHANGED_AUG = ['jigsaw_trans', 'shuffle_trans']
    ENHANCED_AUG = ['hist_brightness_aug', 'adjust_brightness',
                    'shift_image', 'gamma_trans', 'rotate_trans']

    def __init__(self, cap=4000):
        self.dmb = DataMemoryBank(cap=cap)

    def get_bboxes(self, nums=10, img_hw=(256, 256), region_hw=(40, 40)):
        ih, iw = img_hw
        rh, rw = region_hw
        xs = list(set(random.choices(range(ih-rh), k=int(nums*1.5))))[:nums]
        ys = list(set(random.choices(range(iw-rw), k=int(nums*1.5))))[:nums]
        ws = [random.choices(range(rw//4, rw, rw//6), k=1)[0] for i in range(nums)]
        hs = [random.choices(range(rh//4, rh, rh//6), k=1)[0] for i in range(nums)]
        bboxes = [[x, y, h, w] for x, y, h, w in zip(xs, ys, ws, hs)]
        return bboxes

    def recover_trans(self, img_from, img_to, bboxes):
        img_new = img_to.copy()
        for (x, y, dx, dy) in bboxes:
            region = img_from[x:x+dx, y:y+dy]
            img_new[x:x+dx, y:y +
                    dy] = region
        return img_new

    def jigsaw_trans(self, img, nr=(5, 12), nc=(5, 12)):
        img_new = img.copy()
        h, w, c = img.shape
        nr, nc = np.random.randint(*nr), np.random.randint(*nc)
        pr, pc = math.ceil(h/nr)*nr-h, math.ceil(w/nc)*nc-w  # fix to 128
        img_new = cv2.copyMakeBorder(img_new, 0, pr, 0, pc, 0)  # padding
        jigsaw = iaa.Jigsaw(nb_rows=nr, nb_cols=nc,
                            max_steps=(3, 5), allow_pad=False)
        img_new = jigsaw(image=img_new)
        img_new = img_new[:h, :w]
        return img_new

        

    def shuffle_trans(self, img, bboxes=None, thr=0.01, win_size=7, img_ref=None, feat=None):
        # if bboxes is None:
        #     bboxes = self.get_bboxes()
        if img_ref is None:
            img_ref = img.copy()
        ih, iw, ic = img.shape
        img_new = img.copy()
        mask = np.zeros_like(img_new, dtype=np.float32)
        # import pdb;pdb.set_trace()
        # print('shuffle_trans',feat,'shuffle_trans')
        

        def get_patch_feat(img, feat, bbox):
            h, w = img.shape[:2]
            x, y, dx, dy = bbox
            x, y, dx, dy = x/h, y/w, dx/h, dy/w
            
            patch_feat = []
            
            for i in range(100):
                if i not in feat: break
                # print(i,feat[i])
                f = feat[i]
                f_c, f_h, f_w = f.shape
                min_x, min_y = int(x*f_h), int(y*f_w)
                max_x, max_y = int((x+dx)*f_h), int((y+dy)*f_w)
                min_x, min_y = max(0, min_x), max(0, min_y)
                max_x, max_y = min(max_x, f_h), min(max_y, f_w)
                if max_x==min_x or max_y==min_y: break
                patch_f = f[:, min_x:max_x, min_y:max_y]
                patch_feat.append(patch_f)
            return patch_feat
        
        
        def generator(idx,feat=None):
            # print('generator',feat,'generator')
            (x, y, dx, dy) = bboxes[idx]
            patch = img[x:x+dx, y:y+dy].copy()
            patch_ref = img_ref[x:x+dx, y:y+dy]
            patch_ref_feat = get_patch_feat(img, feat, bboxes[idx]) if feat!=None else None
            # print(idx, len(self.dmb.mem),[prf.shape for prf in patch_ref_feat])
            if len(self.dmb.mem) < len(bboxes)*9: # manually
                nx = random.choices(range(1, ih-dx), k=1)[0]
                ny = random.choices(range(1, iw-dy), k=1)[0]
                patch_ref_new = img_ref[nx:nx+dx, ny:ny+dy]
                patch_new = img[nx:nx+dx, ny:ny+dy]
                ssim = self.get_ssim(patch_ref.mean(-1), patch_ref_new.mean(-1), 
                                    win_size=win_size)
                mask_value = 1.0 if ssim<thr else 0.0
            else: # chose patch from memory
                pos, neg = self.dmb.get_pos_neg(patch, npos=2, nneg=2,
                                                use='emb2',
                                                feat=patch_ref_feat,
                                                )
                
                if pos is not None and neg is not None:
                    # ssim = self.get_ssim(neg.mean(-1), patch.mean(-1), 
                    #                 win_size=win_size)
                    patch_new,mask_value = (pos,0.0) if random.random() < 0.5  \
                        else (neg,1.0)
                    if random.random() < 0.01:
                        cv2.imwrite('./sample.png', np.concatenate([patch,pos,neg],axis=1))
                else:
                    patch_new,mask_value = patch, 0
                    
            return patch_ref, patch_ref_feat, patch_new, mask_value, idx
        
        # with Pool(4) as pool:
        # pool = Pool(4)
        # items = pool.map(generator, range(len(bboxes)))
        # pool.close()
        # pool.join()
        for idx in range(len(bboxes)):
            patch_ref, patch_ref_feat, patch_new, mask_value, idx = generator(idx,feat=feat)
        # for patch_ref, patch_new, mask_value, idx in items:
            (x, y, dx, dy) = bboxes[idx]
            img_new[x:x+dx, y:y+dy] = patch_new
            mask[x:x+dx, y:y+dy] = mask_value
            self.dmb.push([patch_ref], feats=[patch_ref_feat])
            

        
        # for (x, y, dx, dy) in bboxes:
        #     if len(self.dmb.mem) < self.dmb.cap: # manually
        #         nx = random.choices(range(1, ih-dx), k=1)[0]
        #         ny = random.choices(range(1, iw-dy), k=1)[0]
        #         img_new[x:x+dx, y:y+dy] = img[nx:nx+dx, ny:ny+dy]
        #         patch = img_ref[x:x+dx, y:y+dy]
        #         patch_new = img_ref[nx:nx+dx, ny:ny+dy]
        #         ssim = self.get_ssim(patch.mean(-1), patch_new.mean(-1), 
        #                             win_size=win_size)
        #         if ssim < thr:
        #             mask[x:x+dx, y:y+dy] = 1.0 
        #         else:
        #             mask[x:x+dx, y:y+dy] = 0.0 
                    
        #         self.dmb.push([patch])
        #     else:
        #         # print('use dmb')
        #         patch = img[x:x+dx, y:y+dy].copy()
        #         if self.dmb.fe is not None:
        #             pos, neg = self.dmb.get_pos_neg(patch, npos=1,nneg=1,use='emb2')
        #         else:
        #             pos, neg = self.dmb.get_pos_neg(patch, npos=1,nneg=1,use='emb1')
        #         if random.random() < 0.5:
        #             img_new[x:x+dx, y:y+dy] = neg
        #             mask[x:x+dx, y:y+dy] = 1.0 if \
        #                 self.get_ssim(neg.mean(-1), patch.mean(-1), win_size=win_size)\
        #                     < thr else 0.0
        #         else:
        #             img_new[x:x+dx, y:y+dy] = pos
        #             mask[x:x+dx, y:y+dy] = 0.0 
        #         # if random.random() < 0.01:
        #         #     cv2.imwrite('./sample.png', np.concatenate([patch,pos,neg],axis=1))
                    
        #     # if random.random()<0.1: print(len(self.dmb.mem))
            
        return img_new, mask
    
    def get_ssim(self, img1, img2, win_size=11):
        ssim = structural_similarity(img1,img2,win_size=win_size)
        return ssim

    def adjust_brightness(self, img, p=0.7, factor_range=(0.2, 0.6)):
        if np.random.random() > p:
            factor = 1.0
        else:
            factor = np.random.uniform(*factor_range)
        img_new = img * factor
        img_new = img_new.astype(img.dtype)
        return img_new

    def shift_image(self, x, dx, dy):
        x = np.roll(x, dy, axis=0)
        x = np.roll(x, dx, axis=1)
        return x

    def hist_brightness_aug(self, img, minv=0.5, maxv=1.5, hwidth=16):
        assert 256 % hwidth == 0, f"256 % hist width ({hwidth}) != 0"

        def get_table():
            return (np.arange(0, 256.0 / 255, 1.0 / 255) ** 
                    np.repeat(np.random.uniform(minv, maxv, size=(256//hwidth,))[..., None], 
                              hwidth, axis=1).flatten() * 255).astype(np.uint8)
        
        table1 = get_table()
        r = cv2.LUT(img[..., 0], table1)
        table2 = get_table()
        g = cv2.LUT(img[..., 1], table2)
        table3 = get_table()
        b = cv2.LUT(img[..., 2], table3)
        img_new = np.stack([r, g, b], 2)
        return img_new

    def gamma_trans(self, img, bboxes, gamma_range=[0, 3]):
        if bboxes is None:
            bboxes = self.get_bboxes()
        img_new = img.copy()
        gamma = np.random.uniform(low=gamma_range[0], high=gamma_range[1])
        img_new = A.functional.gamma_transform(img_new, gamma)
        for (x, y, dx, dy) in bboxes:
            gamma = np.random.uniform(low=gamma_range[0], high=gamma_range[1])
            region = img[x:x+dx, y:y+dy]
            img_new[x:x+dx, y:y +
                    dy] = A.functional.gamma_transform(region, gamma)
        return img_new

    def rotate_trans(self, img, bboxes, rotate_range=[-30, 30]):
        if bboxes is None:
            bboxes = self.get_bboxes()
        img_new = img.copy()
        for (x, y, dx, dy) in bboxes:
            gamma = np.random.randint(
                low=rotate_range[0], high=rotate_range[1])
            region = img[x:x+dx, y:y+dy]
            img_new[x:x+dx, y:y +
                    dy] = rotate(region, gamma, reshape=False)
        return img_new

    def affine_trans(self, img, bboxes):
        raise NotImplementedError
    

    def style_transfer(self, source_image, target_image):
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
