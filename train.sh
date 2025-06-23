# train on SYSU-MM01
python train.py --exp_name SYSU_exp --cfg configs/SYSU.yml --use_rsa --ma_w 0.2 --sample_method camera_random --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 7

# train on RegDB
python train.py --exp_name RegDB_exp --cfg configs/RegDB.yml --use_rsa --ma_w 0.3 --sample_method camera_random  --extra_start_epoch 120 --group_ce_w 1 --group_cs_w 1 --device 7
