import argparse
import os

import cv2
import tqdm
import albumentations as A
import torch
import numpy as np
from torch.utils.data import DataLoader

from build import *
from utils import load_checkpoint, vis_output_by_label, Metric


@torch.no_grad()
def run(args):

    cd_ckpt_path = args.cd_ckpt_path
    save_dir = args.save_dir
    device = args.device
    factor = args.factor
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
                             st=False,
                             )

    test_loader = DataLoader(dataset=test_set,
                             pin_memory=True,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False,
                             drop_last=False,
                             sampler=None)

    metric_val = Metric(init_metric={'f1': 0.0, 'iou': 0.0, 'pr': 0.0, 're': 0.0,})

    # finetune bn
    # model.train()
    # for i in range(3):
    #     for batch, data in enumerate(test_loader):
    #         img1 = data[0].to(device)
    #         img2 = data[1].to(device)
    #         pred = model(img1, img2)
    # model.eval()
    
    # inference
    metric_val.reset()
    for batch, data in tqdm.tqdm(enumerate(test_loader),total=len(test_loader)):
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        label = data[2].to(device)
        pred = model(img1, img2)
        output = pred.argmax(dim=1)
        metric_val(output, label)
        output = output.cpu().numpy()[0]*factor
        save_name = test_set.get_file_name(batch, suffix='.png')
        save_path = os.path.join(save_dir, save_name)
        
        if args.together_show:
            imgs = [img1[0].permute(1,2,0).cpu().numpy()*255,
                    img2[0].permute(1,2,0).cpu().numpy()*255]
            if len(data)>=3:
                label = data[2][0].cpu().numpy()
                output = vis_output_by_label(pred=output,label=label)
                if label.max()==1:
                    label = label * factor
                imgs.append(np.tile(label[:, :, np.newaxis], (1, 1, 3)))
            else:
                output = np.tile(output[:, :, np.newaxis], (1, 1, 3))
            imgs.append(output)
            output = np.concatenate(imgs, axis=1)
            cv2.imwrite(save_path, output)
            
        else:
            cv2.imwrite(save_path, output)
    
    res = metric_val.calculate(local=False)
    pstr = metric_val.print(local=False)
    print(pstr)
        


def main():
    parser = argparse.ArgumentParser(
        description='infer')

    parser.add_argument('-ccp', '--cd_ckpt_path',  type=str, default=None,
                        help='change detection ckpt path')
    parser.add_argument('-sd', '--save_dir', type=str, default=None,
                        help='dir for saving images')
    parser.add_argument('-de', '--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('-f', '--factor', type=int, default=1,
                        help='factor*output in cv2.imwrite')
    parser.add_argument('-dat', '--decoder_attention_type', type=str, default=None,
                        help='decoder attention type')
    parser.add_argument('-ts', '--together-show', action='store_true', default=False,
                        help='together_show_imgs')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
