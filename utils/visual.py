import os
import argparse
import tqdm
import cv2

import numpy as np

def vis_output_by_label(pred, label):
    # 预处理
    pred[pred>0]=1
    label[label>0]=1
    
    # 计算 TP、TN、FP 和 FN 的掩码
    tp_mask = (pred == 1) & (label == 1)
    tn_mask = (pred == 0) & (label == 0)
    fp_mask = (pred == 1) & (label == 0)
    fn_mask = (pred == 0) & (label == 1)

    # 将 TP、TN、FP 和 FN 分别映射到白色、黑色、绿色和红色
    output = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    output[tp_mask] = (255, 255, 255)
    output[tn_mask] = (0, 0, 0)
    output[fp_mask] = (0, 255, 0)
    output[fn_mask] = (255, 0, 0)
    
    return output


def show_images(images, save_path, is_tensor=True):
    if is_tensor:
        # print(images)
        images = [(image*255).detach().cpu().numpy().astype(np.uint8) for image in images]
    images = [image if len(image.shape)>2 else np.stack([image]*3,axis=-1) for image in images]
    vis = np.concatenate(images, axis=1)
    cv2.imwrite(save_path, vis[...,::-1])
    return vis
    


def together_show(args):
    file = args.meta_file
    sr = args.save_root
    dr = args.data_root
    lines = open(file).readlines()
    for line in tqdm.tqdm(lines,total=len(lines)):
        s = line.strip().split(args.sep)[:2]
        paths = [os.path.join(dr, p) for p in s]
        imgs = [cv2.imread(p) for p in paths]
        vis = np.concatenate(imgs, axis=1)
        bname = os.path.basename(paths[-1])
        sname = os.path.join(sr, bname)
        cv2.imwrite(sname, vis)
        

def vis_results(args):
    file = args.meta_file
    sr = args.save_root
    dr = args.data_root
    lines = open(file).readlines()
    for line in tqdm.tqdm(lines,total=len(lines)):
        s = line.strip().split(args.sep)[:3]
        paths = [os.path.join(dr, p) for p in s]
        imgs = [cv2.imread(p) for p in paths[:2]]
        mask = cv2.imread(paths[2],-1)
        mask[mask>0] = 255
        zero = np.zeros_like(mask)
        mask = np.stack([zero,zero,mask],axis=-1)
        vis0 = np.concatenate(imgs,axis=1)
        vis1 = np.concatenate([imgs[0]*0.6+mask*0.4, imgs[1]*0.6+mask*0.4],axis=1)
        vis = np.concatenate([vis0,vis1],axis=0)
        bname = os.path.basename(paths[-1])
        sname = os.path.join(sr, bname)
        cv2.imwrite(sname, vis)




def main():
    parser = argparse.ArgumentParser(
        description='vis')
    parser.add_argument('-mf','--meta_file', type=str, help='path to meta file, such as train.txt and val.txt')
    parser.add_argument('--sep', type=str, default='\t', help='sep in meta file')
    parser.add_argument('-sr','--save_root', type=str, help='where to save vis')
    parser.add_argument('-dr','--data_root', type=str, help='data root in meta file')
    parser.add_argument('-c','--choice', type=int, default=0, help='choice for visualization')
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)

    if args.choice == 0:
        together_show(args)
    elif args.choice == 1:
        vis_results(args)



if __name__=="__main__":
    main()
