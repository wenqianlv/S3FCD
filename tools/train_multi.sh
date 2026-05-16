
# hyper




# export SUFFIX=$( date "+%Y%m%d%H%M")
# python -m torch.distributed.launch --nproc_per_node=2  --master_port=23456 \
# 		 train.py -bs 10 -nw 4 -pp 10 -ne 30 -lr 5e-4 -ft 0.15 -pv 20 -wd ./work_dirs/train/ \
# 		 > ./work_dirs/logs/train_${SUFFIX} 2>&1 







# old
#export SUFFIX=$( date "+%Y%m%d%H%M")
#python -m torch.distributed.launch --nproc_per_node=2  --master_port=23456 \
#		 train.py -bs 10 -nw 4 -pp 10 -ne 25 -lr 1e-3 -ft 0.4 -pv 20 -wd ./work_dirs/train/ \
#		 > ./work_dirs/logs/train_${SUFFIX} 2>&1 

export SUFFIX=$( date "+%Y%m%d%H%M")
python -m torch.distributed.launch --nproc_per_node=2  --master_port=23456 \
		 train.py -bs 10 -nw 4 -pp 10 -ne 25 -lr 5e-4 -ft 0.4 -pv 20 -wd ./work_dirs/train/ \
		 > ./work_dirs/logs/train_${SUFFIX} 2>&1 

export SUFFIX=$( date "+%Y%m%d%H%M")
python -m torch.distributed.launch --nproc_per_node=2  --master_port=23456 \
		 train.py -bs 10 -nw 4 -pp 10 -ne 25 -lr 1e-4 -ft 0.4 -pv 20 -wd ./work_dirs/train/ \
		 > ./work_dirs/logs/train_${SUFFIX} 2>&1 

# export SUFFIX=$( date "+%Y%m%d%H%M")
# python -m torch.distributed.launch --nproc_per_node=2  --master_port=23456 \
#  		 train.py -bs 10 -nw 4 -pp 10 -ne 25 -lr 5e-5 -ft 0.4 -pv 20 -wd ./work_dirs/train/ \
#  		 > ./work_dirs/logs/train_${SUFFIX} 2>&1  &

