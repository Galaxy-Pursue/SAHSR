# python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3

## 论文里面展示的最好效果是这个
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3 # 幽默怎么这个最高？这个是论文里面的结果 r1: 86.26 mAP:83.91