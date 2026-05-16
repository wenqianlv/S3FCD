import argparse
import os

import cv2
import albumentations as A
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
current_path = os.path.abspath(__file__)
code_root = os.path.join(os.path.dirname(current_path), '../')
sys.path.insert(0, code_root)

from build import *
from utils import load_checkpoint, vis_output_by_label



def extract_feat(model, data):
    features = model.encoder(data)
    # import pdb;pdb.set_trace()
    features = {idx:feat[0].detach().cpu().numpy() 
                for idx,feat in enumerate(features)}
    features["info"] = {
        "shape": 'c,h,w',
    }
    return features


def get_save_path(path, data_root, save_root, suffix='.npy'):
    # import pdb;pdb.set_trace()
    path = path.replace(data_root, save_root)
    save_path = path[:path.rfind('.')]+suffix
    if not os.path.exists(os.path.dirname(save_path)):
        os.system(f"mkdir -p {os.path.dirname(save_path)}")
    return save_path

@torch.no_grad()
def run(args):

    cd_ckpt_path = args.cd_ckpt_path
    save_dir = args.save_dir
    device = args.device
    decoder_attention_type = args.decoder_attention_type

    os.system(f"mkdir -p {save_dir}")

    device = torch.device(device)

    # model
    model = build_model(choice='cdp_UnetPlusPlus', encoder_name="timm-efficientnet-b0",
                        encoder_weights="noisy-student",
                        decoder_attention_type=decoder_attention_type,
                        in_channels=3,
                        classes=2,
                        siam_encoder=True,
                        fusion_form='concat',
                        )
    model = model.to(device)
    load_checkpoint(cd_ckpt_path, {"state_dict": model})
    model.eval()

    # pipeline
    test_pipeline = A.Compose([
        A.HorizontalFlip(p=0.0), ],
        additional_targets={'image1': 'image'})

    # dataloader
    test_set = build_dataset(choice='CommonDataset',
                             data_root="/root/autodl-tmp/dataset/CLCD",
                             metafile="/root/autodl-tmp/dataset/CLCD/val.txt",
                             test_mode=False,
                             pipeline=test_pipeline,
                             c255t1_in_mask=True,
                             )

    test_loader = DataLoader(dataset=test_set,
                             pin_memory=True,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False,
                             drop_last=False,
                             sampler=None)
    
    # inference
    for batch, data in tqdm(enumerate(test_loader),total=len(test_loader)):
        img1 = data[0].to(device)
        img2 = data[1].to(device)

        feat1 = extract_feat(model, img1)
        feat2 = extract_feat(model, img2)
        
        item = test_loader.dataset.get_single_item(batch)
        path1, path2 = item[:2]
        spath1 = get_save_path(path1,"/root/autodl-tmp/dataset/CLCD",save_dir)
        spath2 = get_save_path(path2,"/root/autodl-tmp/dataset/CLCD",save_dir)
        
        np.save(spath1, feat1)
        np.save(spath2, feat2)
        
        # break
        


def main():
    parser = argparse.ArgumentParser(
        description='infer')

    parser.add_argument('-ccp', '--cd_ckpt_path',  type=str, default=None,
                        help='change detection ckpt path')
    parser.add_argument('-sd', '--save_dir', type=str, default=None,
                        help='dir for saving features')
    parser.add_argument('-de', '--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('-dat', '--decoder_attention_type', type=str, default=None,
                        help='decoder attention type')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
