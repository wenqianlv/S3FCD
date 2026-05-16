SUFFIX:=$(shell date "+%Y%m%d%H%M")
env:
	pip install -r requirements.txt

init:
	mkdir -p ./work_dirs
	cd ./work_dirs && \
		mkdir logs debug_cpu debug_gpu train infer

init_sysu:
	echo "init sysu"
	mkdir -p /root/autodl-tmp/dataset/SYSU-CD/
	cp -r /root/autodl-nas/dataset/SYSU-CD/* /root/autodl-tmp/dataset/SYSU-CD/
	cd /root/autodl-tmp/dataset/SYSU-CD && \
		unzip train.zip && \
		unzip val.zip && \
		unzip test.zip && \
		rm train.zip val.zip test.zip

debug_cpu:
	python3 train.py -bs 1 -nw 0 -pp 10 --ema -wd ./work_dirs/debug_cpu/

debug_gpu:
	python3 -m torch.distributed.launch --nproc_per_node=1  --master_port=23456 \
		 train.py -bs 4 -nw 1 -pp 10 -lr 1e-4 -pv 50 -wd ./work_dirs/debug_gpu/

train:
	python -m torch.distributed.launch --nproc_per_node=2  --master_port=23456 \
 		 train.py -bs 10 -nw 4 -pp 10 -ne 25 -lr 5e-4 -ft 0.4 -pv 20 -wd ./work_dirs/train/ \
 		 > ./work_dirs/logs/train_${SUFFIX} 2>&1  &
	tail -f ./work_dirs/logs/train_${SUFFIX}

infer:
	python infer.py -ccp \
			/root/autodl-nas/code/Lab0330/configs/230227_base_w10r5_dmb/work_dirs/train/epoch_24_f1_0.602_iou_0.431_pr_0.546_re_0.672.pth \
		-sd work_dirs/infer/230227_base_w10r5_dmb_on_s2looking_${SUFFIX} \
		-f 255 \
		-ts 

val:
	python val.py -ccp \
			/root/autodl-nas/code/Lab0330/configs/230318_base_w10r5_dmb_clcd_train/work_dirs/train/epoch_25_f1_0.395_iou_0.246_pr_0.322_re_0.513.pth \
		-sd work_dirs/val/230318_base_w10r5_dmb_clcd_train_${SUFFIX} \
		-f 255 \
		-ts \
		> ./work_dirs/logs/val_${SUFFIX} 2>&1 &
	tail -f ./work_dirs/logs/val_${SUFFIX}

val_ss:
	python val.py -ccp \
			/root/autodl-nas/code/Lab0330/configs/230315_supervised_w10r5_dmb_train/work_dirs/train/epoch_16_f1_0.792_iou_0.655_pr_0.801_re_0.782.pth \
		-sd work_dirs/val/230315_supervised_w10r5_dmb_train \
		-f 255 \
		-ts \
		> ./work_dirs/logs/val_${SUFFIX} 2>&1 &
	tail -f ./work_dirs/logs/val_${SUFFIX}


extract_feat:
	python tools/extract_feat.py -ccp \
			/root/autodl-nas/code/Lab0330/configs/230323_base_w10r5_dmb_clcd_train_model/work_dirs/train/epoch_10_f1_0.462_iou_0.300_pr_0.424_re_0.507.pth \
		-sd work_dirs/feat/230315_supervised_w10r5_dmb_train \

extract_feat_val:
	python tools/extract_feat.py -ccp \
			/root/autodl-nas/code/Lab0330/configs/230323_base_w10r5_dmb_clcd_train_model/work_dirs/train/epoch_10_f1_0.462_iou_0.300_pr_0.424_re_0.507.pth \
		-sd work_dirs/feat/230315_supervised_w10r5_dmb_val \

clean:
	rm -rf ./work_dirs/*/*

kill_train:
	ps aux | grep train | awk '{print $$2}' | xargs kill -s 9

