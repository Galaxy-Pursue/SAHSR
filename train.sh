python train.py --exp_name my_inforce_behindpp_inforcew1_projection2048dim --inforce_w 1 --dp_w 0.3 --cs_w 1 --device 0
python train.py --exp_name my_inforce_behindpp_inforcew1e-1_projection2048dim --inforce_w 0.1 --dp_w 0.3 --cs_w 1 --device 1
python train.py --exp_name my_inforce_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 2
python train.py --exp_name my_inforce_behindpp_inforcew1e-3_projection2048dim --inforce_w 0.001 --dp_w 0.3 --cs_w 1 --device 3
python train.py --exp_name no_inforce --inforce_w 0 --dp_w 0.3 --cs_w 1 --device 4

python train.py --exp_name vi_inforce_behindpp_inforcew1_projection2048dim --inforce_w 1 --dp_w 0.3 --cs_w 1 --device 0 --inforce_type new
python train.py --exp_name vi_inforce_behindpp_inforcew1e-1_projection2048dim --inforce_w 0.1 --dp_w 0.3 --cs_w 1 --device 1 --inforce_type new
python train.py --exp_name vi_inforce_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 2 --inforce_type new
python train.py --exp_name vi_inforce_behindpp_inforcew1e-3_projection2048dim --inforce_w 0.001 --dp_w 0.3 --cs_w 1 --device 3 --inforce_type new
python train.py --exp_name vi_inforce_behindpp_inforcew1e-4_projection2048dim --inforce_w 0.0001 --dp_w 0.3 --cs_w 1 --device 5 --inforce_type new
python train.py --exp_name vi_inforce_behindpp_inforcew1e-5_projection2048dim --inforce_w 0.00001 --dp_w 0.3 --cs_w 1 --device 6 --inforce_type new

python train.py --exp_name retrain_my_inforce_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 0
python train.py --exp_name retrain_vi_inforce_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 1 --inforce_type new
python train.py --exp_name retrian_baseline_no_inforce --inforce_w 0 --dp_w 0.3 --cs_w 1 --device 2

python train.py --exp_name my_inforce_behindpp_inforcew1e-2_no_projection --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 3
python train.py --exp_name vi_inforce_behindpp_inforcew1e-2_no_projection --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 4 --inforce_type new

python train.py --exp_name continue_train_my_inforce_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 0 --start_train_epoch 160 --train_epoch 240 --resume /home/lky/vi525/logs/inforce/retrain_my_inforce_behindpp_inforcew1e-2_projection2048dim/checkpoints/model_best.pth
python train.py --exp_name continue_train_vi_inforce_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 1 --inforce_type new --start_train_epoch 160 --train_epoch 240 --resume /home/lky/vi525/logs/inforce/retrain_vi_inforce_behindpp_inforcew1e-2_projection2048dim/checkpoints/model_best.pth
python train.py --exp_name continue_train_baseline_no_inforce --inforce_w 0 --dp_w 0.3 --cs_w 1 --device 2 --start_train_epoch 160 --train_epoch 240 --resume /home/lky/vi525/logs/inforce/retrian_baseline_no_inforce/checkpoints/model_best.pth

python train.py --exp_name baseline_debugcs --inforce_w 0 --dp_w 0.3 --cs_w 1 --device 0 --cs_type debug
python train.py --exp_name baseline_debugcs_inforce_old_inforcew1e-2 --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 1 --cs_type debug --inforce_type old
python train.py --exp_name baseline_debugcs_viinforce_inforcew1e-2 --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 2 --cs_type debug --inforce_type new


python train.py --exp_name my_inforce_behindpp_inforcew1e-1_projection2048dim_simtype_new --inforce_w 0.1 --dp_w 0.3 --cs_w 1 --device 0 --inforce_sim_type cos_sim
python train.py --exp_name my_inforce_behindpp_inforcew1e-2_projection2048dim_simtype_new --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 1 --inforce_sim_type cos_sim
python train.py --exp_name my_inforce_behindpp_inforcew1e-3_projection2048dim_simtype_new --inforce_w 0.001 --dp_w 0.3 --cs_w 1 --device 2 --inforce_sim_type cos_sim

python train.py --exp_name vi_inforce_behindpp_inforcew1e-1_projection2048dim_simtype_new --inforce_w 0.1 --dp_w 0.3 --cs_w 1 --device 4 --inforce_type new --inforce_sim_type cos_sim
python train.py --exp_name vi_inforce_behindpp_inforcew1e-2_projection2048dim_simtype_new --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 5 --inforce_type new --inforce_sim_type cos_sim
python train.py --exp_name vi_inforce_behindpp_inforcew1e-3_projection2048dim_simtype_new --inforce_w 0.001 --dp_w 0.3 --cs_w 1 --device 6 --inforce_type new --inforce_sim_type cos_sim

python train.py --exp_name centerNCE_behindpp_inforcew1e-1_projection2048dim --inforce_w 0.1 --dp_w 0.3 --cs_w 1 --device 0 --inforce_type centerNCE
python train.py --exp_name centerNCE_behindpp_inforcew1e-2_projection2048dim --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 1 --inforce_type centerNCE
python train.py --exp_name centerNCE_behindpp_inforcew1e-3_projection2048dim --inforce_w 0.001 --dp_w 0.3 --cs_w 1 --device 2 --inforce_type centerNCE

python train.py --exp_name centerNCE_behindpp_inforcew1e-1_projection2048dim_normalized --inforce_w 0.1 --dp_w 0.3 --cs_w 1 --device 4 --inforce_type centerNCE
python train.py --exp_name centerNCE_behindpp_inforcew1e-2_projection2048dim_normalized --inforce_w 0.01 --dp_w 0.3 --cs_w 1 --device 5 --inforce_type centerNCE
python train.py --exp_name centerNCE_behindpp_inforcew1e-3_projection2048dim_normalized --inforce_w 0.001 --dp_w 0.3 --cs_w 1 --device 6 --inforce_type centerNCE

python train.py --exp_name baseline_no_cs_no_inforce --inforce_w 0 --dp_w 0.3 --cs_w 0 --device 7


python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 0 --cs1_type dist_pc_loss
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-1 --inforce_w 0 --dp_w 0.3 --cs1_w 0.1 --device 1 --cs1_type dist_pc_loss
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-2 --inforce_w 0 --dp_w 0.3 --cs1_w 0.01 --device 2 --cs1_type dist_pc_loss
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-3 --inforce_w 0 --dp_w 0.3 --cs1_w 0.001 --device 3 --cs1_type dist_pc_loss

python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1_projection --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 0 --cs1_type dist_pc_loss
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-1_projection --inforce_w 0 --dp_w 0.3 --cs1_w 0.1 --device 1 --cs1_type dist_pc_loss
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-2_projection --inforce_w 0 --dp_w 0.3 --cs1_w 0.01 --device 2 --cs1_type dist_pc_loss
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-3_projection --inforce_w 0 --dp_w 0.3 --cs1_w 0.001 --device 3 --cs1_type dist_pc_loss

python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1_projection --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 4 --cs1_type centerNCE
python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1e-1_projection --inforce_w 0 --dp_w 0.3 --cs1_w 0.1 --device 5 --cs1_type centerNCE
python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1e-2_projection --inforce_w 0 --dp_w 0.3 --cs1_w 0.01 --device 6 --cs1_type centerNCE
python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1e-3_projection --inforce_w 0 --dp_w 0.3 --cs1_w 0.001 --device 7 --cs1_type centerNCE

# 6.17 实验ppool3+w/o ada条件下cs1使用dist_pc和centerNCE的效果对比，考虑之后彻底去掉projection的愚蠢设计
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 0 --cs1_type dist_pc_loss --ppool_type ppool3
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-1_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 0.1 --device 2 --cs1_type dist_pc_loss --ppool_type ppool3
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1e-2_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 0.01 --device 3 --cs1_type dist_pc_loss --ppool_type ppool3

python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 4 --cs1_type centerNCE --ppool_type ppool3
python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1e-1_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 0.1 --device 5 --cs1_type centerNCE --ppool_type ppool3
python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1e-2_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 0.01 --device 6 --cs1_type centerNCE --ppool_type ppool3

## 再补一组对比实验，ppool3+w/o ada条件下的baseline
python train.py --exp_name baseline_no_inforce_no_cs_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 0 --device 7 --ppool_type ppool3

python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1_cs2_centerNCE_cs2w1_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 1  --cs2_w 1 --device 0 --cs1_type dist_pc_loss --cs2_type centerNCE --ppool_type ppool3
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1_cs2_centerNCE_cs2w1e-1_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 1 --cs2_w 0.1 --device 1 --cs1_type dist_pc_loss --cs2_type centerNCE --ppool_type ppool3
python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1_cs2_centerNCE_cs2w1e-2_ppool3_woada --inforce_w 0 --dp_w 0.3 --cs1_w 1 --cs2_w 0.01 --device 2 --cs1_type dist_pc_loss --cs2_type centerNCE --ppool_type ppool3

python train.py --exp_name baseline_no_inforce_cs1_distpc_cs1w1_ppool3_ada --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 3 --cs1_type dist_pc_loss --ppool_type ppool3 --use_ada
python train.py --exp_name baseline_no_inforce_cs1_centerNCE_cs1w1_ppool3_ada --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 4 --cs1_type centerNCE --ppool_type ppool3 --use_ada


# 6.21 在ppool2+ada+w/o projection+w/o inforce+cs1_w=1的条件下，测试加入margin是否可以避免centerNCE过拟合，从而提升模型的性能
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 0 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um095_dm-1 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 2 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.95 --down_margin -1
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um09_dm-1 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 3 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.9 --down_margin -1
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um1_dm-095 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 4 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 1 --down_margin -0.95
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um095_dm-095 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 5 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.95 --down_margin -0.95
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um09_dm-095 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 6 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.9 --down_margin -0.95
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um07_dm-07 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 7 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.7 --down_margin -0.7
## margin设置的太大一点用都没有，除了导数第一组实验其他实验的效果完全一样

# 6.24 在ppool2+ada+w/o projection+w/o inforce+cs1_w=1的条件下，测试加入margin是否可以避免centerNCE过拟合，从而提升模型的性能
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um065_dm-065 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 2 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.65 --down_margin -0.65
python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um075_dm-075 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 5 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.75 --down_margin -0.75

python train.py --exp_name baseline_no_inforce_cs1_margincenterNCE_cs1w1_ppool2_ada_noprojection_um07_dm-065 --inforce_w 0 --dp_w 0.3 --cs1_w 1 --device 2 --cs1_type margin_centerNCE --ppool_type ppool2 --use_ada --up_margin 0.7 --down_margin -0.65


python train.py --exp_name base8_cbs_woinforce --device 0
python train.py --exp_name base8_cbs_centerNCE1e-3 --device 3 --inforce_w 0.001
python train.py --exp_name base8_cbs_centerNCE1e-2 --device 4 --inforce_w 0.01
python train.py --exp_name base8_cbs_centerNCE1e-1 --device 5 --inforce_w 0.1
python train.py --exp_name base8_cbs_centerNCE1 --device 7 --inforce_w 1

# extra_sampler
# 我傻了，这些实验没有用stn
python train.py --exp_name base8_cbs_extra_woinforce_start0 --device 0 --extra_start_epoch 0
python train.py --exp_name base8_cbs_woinforce_start40 --device 1 --extra_start_epoch 40  # 名字写错了实际上是extra
python train.py --exp_name base8_cbs_woinforce_start80 --device 2 --extra_start_epoch 80
python train.py --exp_name base8_cbs_woinforce_start120 --device 3 --extra_start_epoch 120
python train.py --exp_name base8_cbs_extra_woinforce_start200 --device 7 --extra_start_epoch 200

# 我傻了，这些实验没有用stn
python train.py --exp_name base8_regdb_cbs_extra_start0 --device 0 --cfg models/model_base8/RegDB.yml --extra_start_epoch 0
python train.py --exp_name base8_regdb_cbs_extra_start40 --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 40
python train.py --exp_name base8_regdb_cbs_extra_start80 --device 2 --cfg models/model_base8/RegDB.yml --extra_start_epoch 80
python train.py --exp_name base8_regdb_cbs_extra_start120 --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120

# 看看改了采样器之后是不是快一点了
python train.py --exp_name base8_sysu_cbs_extra_start0_testspeed --device 0 --cfg configs/SYSU.yml --extra_start_epoch 0
python train.py --exp_name base8_regdb_cbs_extra_start0_testspeed2 --device 5 --cfg models/model_base8/RegDB.yml --extra_start_epoch 0

python train.py --exp_name base8_sysu_cbs_extra_start120_stn --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120
python train.py --exp_name base8_sysu_cbs_extra_start0_testspeed --device 6 --cfg configs/SYSU.yml --extra_start_epoch 0

python train.py --exp_name base8_sysu_cbs_extra_start0_stn --device 4 --cfg configs/SYSU.yml --extra_start_epoch 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn --device 5 --cfg configs/SYSU.yml --extra_start_epoch 40
python train.py --exp_name base8_sysu_cbs_extra_start80_stn --device 6 --cfg configs/SYSU.yml --extra_start_epoch 80


# 将自身id加入到extra训练中
# 奇怪了怎么和上面的结果一模一样？？？
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_addself --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120
python train.py --exp_name base8_sysu_cbs_extra_start0_stn_addself --device 7 --cfg configs/SYSU.yml --extra_start_epoch 0
#python train.py --exp_name base8_sysu_cbs_extra_start80_stn_addself --device 6 --cfg configs/SYSU.yml --extra_start_epoch 80

# retrain extra+stn+addself
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_retrain --device 5 --cfg configs/SYSU.yml --extra_start_epoch 120
python train.py --exp_name base8_sysu_cbs_extra_start80_stn_addself --device 6 --cfg configs/SYSU.yml --extra_start_epoch 80

# 7.29 实验extra_train_loss针对性区分具有高相似性的样本
python train.py --exp_name base8_sysu_cbs_extra_start0_stn_addself_extraloss --device 3 --cfg configs/SYSU.yml --extra_start_epoch 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_addself_extraloss --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40
python train.py --exp_name base8_sysu_cbs_extra_start80_stn_addself_extraloss --device 5 --cfg configs/SYSU.yml --extra_start_epoch 80
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120

# 7.30对于额外loss使用一个新的优化器对模型参数进行优化
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_00005 --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_00001 --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00001
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_000005 --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.000005
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_000001 --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.000001

python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_00005_continue --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_00001_continue --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00001 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth

# 7.31 对上述实验进行修改以提升性能
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_extralr0_00005_fullce_continue --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth

# 麻了代码写错了这三组实验无效
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_001inforce_extralr0_00005_continue --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.01
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_1inforce_extralr0_00005_continue --device 5 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 1
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_0001inforce_extralr0_00005_continue --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.001


# 我又麻了忘了把inforce的参数加进去导致上面三组实验又白跑了
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_1inforce_extralr0_00005_continue --device 5 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 1
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_001inforce_extralr0_00005_continue --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.01
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_0001inforce_extralr0_00005_continue --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.001

# 实验一下增大额外优化器的学习率会不会效果好一点
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_001inforce_extralr0_0001_continue --device 6 --cfg configs/SYSU.yml --extra_start_epoch 120 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.01 --lr_extra 0.0001
# 试试再加上csloss会不会有效果提升
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_001gpinforce_1gpcs_extralr0_00005_continue --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.01 --group_cs_w 1
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_addself_extraloss_ce_001gpinforce_01gpcs_extralr0_00005_continue --device 5 --cfg configs/SYSU.yml --extra_start_epoch 120 --lr_extra 0.00005 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --group_inforce_w 0.01 --group_cs_w 0.1

# 8.2 效果仍然不太理想，尝试提前开启extra_train看看是否会有效果提升
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_addself_extraloss_ce_001gpinforce_1gpcs_extralr0_00005 --device 7 --cfg configs/SYSU.yml --extra_start_epoch 40 --lr_extra 0.00005  --group_inforce_w 0.01 --group_cs_w 0.1
## 不使用extra_loss，只使用原来的loss进行额外sampler的训练，对extra_sampler使用单独的optimizer，学习率设定为5e-5，start120
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_woextraloss_extralr0_00005_continue --device 7 --cfg configs/SYSU.yml --extra_start_epoch 120 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --lr_extra 0.00005
## 不使用extra_loss，只使用原来的loss进行额外sampler的训练，对extra_sampler使用单独的optimizer，学习率设定为0.00035(同原始optimizer)，start120
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_woextraloss_extralr0.00035_continue --device 4 --cfg configs/SYSU.yml --extra_start_epoch 120 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth --lr_extra 0.00035
## 不使用extra_loss，只使用原来的loss进行额外sampler的训练，对extra_sampler使用原来的optimizer，start120
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_woextraloss_sameopti_continue --device 1 --cfg configs/SYSU.yml --extra_start_epoch 120 --start_train_epoch 118 --train_epoch 160 --resume ./logs/model_base8/base8_118epoch_stn_addself_extrastart120.pth
## 不使用extra_loss，只使用原来的loss进行额外sampler的训练，对extra_sampler使用原来的optimizer，start40
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_woextraloss_sameopti --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40

# 8.3 目前效果最好的一版是第188行的实验（cbs+extra sampler 40start，不设置单独的extra_loss和额外的优化器）
## 为了能水创新点，在原有的loss上面加入一个权重很小的extra_loss
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_sameopti --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-2_sameopti --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.01
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-4_sameopti --device 7 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.0001



python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_sameopti_testresult --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --start_train_epoch 118 --train_epoch 160 --resume

# 8.4 加了extra_loss效果不如不加，我试试多加一点和少加一点
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-5_sameopti --device 7 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.00001
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-2_sameopti --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.01
## 下面这个是最好的
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001

# 做一组只使用sampler，不使用stn和ppool的消融实验，记得把basline.py的参数改回去
python train.py --exp_name base8_sysu_cbs_extra_start40_wostn_woppool_additionalextraloss_gpcew1e-3_sameopti --device 2 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001

## 奇了怪了和另一台电脑上的结果相差很大，重新跑一遍
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_sameopti_retrain --device 1 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_retrain --device 2 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001


# 8.8 测试在regdb上模型的效果
python train.py --exp_name base8_regdb_cbs_extra_start150_gcew1e-3 --device 0 --cfg models/model_base8/RegDB.yml --extra_start_epoch 150 --group_ce_w 0.001
python train.py --exp_name base8_regdb_cbs_extra_start161 --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 161
python train.py --exp_name base8_regdb_cbs_extra_start140_gcew1e-3 --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 140 --group_ce_w 0.001
python train.py --exp_name base8_regdb_cbs_extra_start135_gcew1e-3 --device 7 --cfg models/model_base8/RegDB.yml --extra_start_epoch 135 --group_ce_w 0.001

python train.py --exp_name base8_regdb_cbs_extra_start161_3e-1pc --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 161
python train.py --exp_name base8_regdb_cbs_extra_start140_gcew1e-3_5e-1pc --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 140 --group_ce_w 0.001
python train.py --exp_name base8_regdb_cbs_extra_start161_5e-1pc --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 161
python train.py --exp_name base8_regdb_cbs_extra_start161_5e-1pc_dpw3e-1 --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 161 --dp_w 0.3

# 8.9 继续在regdb上调参测试CHSR，记得dp_w调整为0.3
## 先测试extra_start_epoch的影响，不使用extra_loss
python train.py --exp_name base8_regdb_cbs_extra_start150_5e-1pc_3e-1dpw_woextraloss --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 150 --dp_w 0.3
python train.py --exp_name base8_regdb_cbs_extra_start140_5e-1pc_3e-1dpw_woextraloss --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 140 --dp_w 0.3
python train.py --exp_name base8_regdb_cbs_extra_start130_5e-1pc_3e-1dpw_woextraloss --device 5 --cfg models/model_base8/RegDB.yml --extra_start_epoch 130 --dp_w 0.3
python train.py --exp_name base8_regdb_cbs_extra_start120_5e-1pc_3e-1dpw_woextraloss --device 6 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.3


# 8.10 sysu上的灵敏性测试
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw5e-1_dpw1e-1 --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.5 --dp_w 0.1
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw5e-1_dpw3e-1 --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.5 --dp_w 0.3
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw5e-1_dpw5e-1 --device 7 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.5 --dp_w 0.5
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw5e-1_dpw7e-1 --device 0 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.5 --dp_w 0.7



python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw5e-1_dpw9e-1 --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.5 --dp_w 0.9


python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw1e-1_dpw7e-1 --device 0 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.1 --dp_w 0.7
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw0.3_dpw0.7 --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.3 --dp_w 0.7
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw0.05_dpw0.7 --device 7 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --pc_w 0.05 --dp_w 0.7


python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_pcw1e-3_ --device 2 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001


# 8.11 跑一下消融实验，对于模块stn和rsc的使用，值为0表示不使用，值为1表示使用
## vbs（200epoch）
python train.py --exp_name base8_sysu_vbs_200epoch_wostn_worsc_woextra_dpw7e-1 --sample_method camera_random --cfg ./models/model_base8/ablation_cfg/vbs200epoch.yml --use_stn 0 --use_rsc 0 --extra_start_epoch 114514 --device 0 --dp_w 0.7
python train.py --exp_name base8_sysu_vbs_200epoch_wostn_worsc_woextra_dpw3e-1 --sample_method camera_random --cfg ./models/model_base8/ablation_cfg/vbs200epoch.yml --use_stn 0 --use_rsc 0 --extra_start_epoch 114514 --device 0 --dp_w 0.3
## chrs（160）
python train.py --exp_name base8_sysu_wovbs_160epoch_wostn_worsc_extra_start40_gce1e-3_gcs1e-3_dpw7e-1 --sample_method identity_random --cfg ./models/model_base8/ablation_cfg/chrs160epoch.yml --use_stn 0 --use_rsc 0 --extra_start_epoch 40 --device 3 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.7
## aafl+vbs（200epoch）
python train.py --exp_name base8_sysu_vbs_200epoch_stn_worsc_woextra_dpw7e-1 --sample_method camera_random --cfg ./models/model_base8/ablation_cfg/aafl_vbs200epoch.yml --use_stn 1 --use_rsc 0 --extra_start_epoch 114514 --device 4 --dp_w 0.7
python train.py --exp_name base8_sysu_vbs_200epoch_stn_worsc_woextra_dpw3e-1 --sample_method camera_random --cfg ./models/model_base8/ablation_cfg/aafl_vbs200epoch.yml --use_stn 1 --use_rsc 0 --extra_start_epoch 114514 --device 4 --dp_w 0.3
## vbs+chrs（160）
python train.py --exp_name base8_sysu_vbs_160epoch_wostn_worsc_extra_start40_gce1e-3_gcs1e-3_dpw7e-1 --sample_method camera_random --cfg ./models/model_base8/ablation_cfg/vbs_chrs160epoch.yml --use_stn 0 --use_rsc 0 --extra_start_epoch 40 --device 7 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.7


# 8.12
## 跑一下不加pc的dpw曲线
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw5e-1 --device 2 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.5 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw7e-1 --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.7 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw3e-1 --device 4 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.3 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw1e-1 --device 5 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.1 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw0 --device 6 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0 --pc_w 0

## 有空跑一下vbs+aafl+dpw0.5
python train.py --exp_name base8_sysu_vbs_200epoch_stn_worsc_woextra_dpw5e-1 --sample_method camera_random --cfg ./models/model_base8/ablation_cfg/aafl_vbs200epoch.yml --use_stn 1 --use_rsc 0 --extra_start_epoch 114514 --device 7 --dp_w 0.5 --pc_w 0

## baseline+aafl降低
python train.py --exp_name base8_sysu_wovbs_160epoch_stn_worsc_woextra_dpw3e-1 --sample_method identity_random --cfg ./models/model_base8/ablation_cfg/aafl160epoch.yml --use_stn 1 --use_rsc 0 --extra_start_epoch 114514 --device 0 --dp_w 0.3 --pc_w 0
python train.py --exp_name base8_sysu_wovbs_160epoch_stn_worsc_woextra_dpw5e-1 --sample_method identity_random --cfg ./models/model_base8/ablation_cfg/aafl160epoch.yml --use_stn 1 --use_rsc 0 --extra_start_epoch 114514 --device 1 --dp_w 0.5 --pc_w 0

# 8.14
## 跑start_epoch的曲线
python train.py --exp_name base8_sysu_cbs_extra_start0_stn_rsc_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw5e-1 --device 2 --cfg configs/SYSU.yml --extra_start_epoch 0 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.5 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start80_stn_rsc_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw5e-1 --device 2 --cfg configs/SYSU.yml --extra_start_epoch 80 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.5 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start120_stn_rsc_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw5e-1 --device 2 --cfg configs/SYSU.yml --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.5 --pc_w 0



python train.py --exp_name base8_sysu_cbs_extra_start40_stn_rsc_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw2e-1 --device 5 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.2 --pc_w 0
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_rsc_additionalextraloss_gpcew1e-3_gpcsw1e-3_dpw4e-1 --device 6 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --dp_w 0.4 --pc_w 0


# 8.15
## 在regdb上调个sota
### 先跑一组不带grouploss的
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start150_5e-1dpw_woextraloss --device 0 --cfg models/model_base8/RegDB.yml --extra_start_epoch 150 --dp_w 0.5
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start140_5e-1dpw_woextraloss --device 1 --cfg models/model_base8/RegDB.yml --extra_start_epoch 140 --dp_w 0.5
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start130_5e-1dpw_woextraloss --device 2 --cfg models/model_base8/RegDB.yml --extra_start_epoch 130 --dp_w 0.5
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start120_5e-1dpw_woextraloss --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.5

python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start150_5e-1dpw_gpcew1e-3_gpcsw1e-3 --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 150 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start140_5e-1dpw_gpcew1e-3_gpcsw1e-3 --device 5 --cfg models/model_base8/RegDB.yml --extra_start_epoch 140 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start130_5e-1dpw_gpcew1e-3_gpcsw1e-3 --device 6 --cfg models/model_base8/RegDB.yml --extra_start_epoch 130 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001
python train.py --exp_name base8_regdb_cbs_rsc_stn_extra_start120_5e-1dpw_gpcew1e-3_gpcsw1e-3 --device 7 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001

### 先看看不用pc和extra单独调dp能调到多少
python train.py --exp_name base8_regdb_cbs_rsc_stn_woextra_5e-1dpw --device 0 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --dp_w 0.5
python train.py --exp_name base8_regdb_cbs_rsc_stn_woextra_3e-1dpw --device 1 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --dp_w 0.3
python train.py --exp_name base8_regdb_cbs_rsc_stn_woextra_1e-1dpw --device 2 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --dp_w 0.1
python train.py --exp_name base8_regdb_cbs_rsc_stn_woextra_7e-1dpw --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --dp_w 0.7

python train.py --exp_name occupy3 --device 5  --extra_start_epoch 14545 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001 --train_epoch 114514
python train.py --exp_name occupy1 --device 6  --extra_start_epoch 14545 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001 --extra_start_epoch 40
python train.py --exp_name occupy2 --device 7  --extra_start_epoch 11465 --dp_w 0.5 --group_ce_w 0.001 --group_cs_w 0.001

### 增大batchsize，看看效果会不会好:好吧效果非常差
python train.py --exp_name base8_regdb_cbs_rsc_stn_woextra_1e-1dpw_144batch --device 0 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --dp_w 0.1
python train.py --exp_name base8_regdb_cbs_rsc_stn_woextra_5e-2dpw_144batch --device 1 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --dp_w 0.05
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_144batch_topk4_start120 --device 2 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 4
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_144batch_topk2_start120 --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 2
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_144batch_topk6_start120 --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 6

### 测试不同topk的影响
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_topk4_start120 --device 2 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 4
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_topk2_start120 --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 2
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_topk6_start120 --device 4 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 6

python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_topk4_start120_gpcew1e-3_gpcsw1e-3 --device 3 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 4 --group_ce_w 0.001 --group_cs_w 0.001
python train.py --exp_name base8_regdb_cbs_rsc_stn_5e-2dpw_topk4_start120_gpcew1e-3_gpcsw1e-3 --device 1 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.05 --top_k 4 --group_ce_w 0.001 --group_cs_w 0.001
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_topk4_start120_gpcew1e-2_gpcsw1e-2 --device 2 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 4 --group_ce_w 0.01 --group_cs_w 0.01
python train.py --exp_name base8_regdb_cbs_rsc_stn_1e-1dpw_topk4_start120_gpcew1e-1_gpcsw1e-1 --device 5 --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --dp_w 0.1 --top_k 4 --group_ce_w 0.1 --group_cs_w 0.1



python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti_samllbatch --device 2 --cfg configs/SYSU_small.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001


# 9.3 vbs+rsa+chrs，全村的希望就靠你了
python train.py --exp_name base8_sysu_cbs_extra_start40_stn_additionalextraloss_gpcew1e-3_gpcsw1e-3_sameopti --device 3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3 # the best
python train.py --exp_name base8_sysu_VBS_RSA_pcw0_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 5
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 6
python train.py --exp_name base8_sysu_VBS_STN_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 1 --dp_w 0.5 --use_SEM --cfg configs/SYSU.yml --pc_w 0.3 --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 7



# 9.4 消融实验
python train.py --exp_name base8_sysu_baseline --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method identity_random --device 7
python train.py --exp_name base8_sysu_baseline_CHRS --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method identity_random --device 7
python train.py --exp_name base8_sysu_baseline_RSA --use_stn 0 --use_rsc 1 --pc_w 0.3  --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method identity_random --device 7
python train.py --exp_name base8_sysu_baseline_RSA_VBS --use_stn 0 --use_rsc 1 --pc_w 0.3  --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method camera_random --device 5
python train.py --exp_name base8_sysu_baseline_RSA_CHRS --use_stn 0 --use_rsc 1 --pc_w 0.3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method identity_random --device 6
python train.py --exp_name base8_sysu_baseline_VBS_CHRS --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method camera_random --device 7

python train.py --exp_name base8_sysu_baseline_VBS --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method camera_random --device 0


python train.py --exp_name base8_sysu_baseline_luppretrained --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method identity_random --LUP_pretrained --device 3



python train.py --exp_name base8_sysu_baseline_RSA_CHRS_SEM --use_SEM --use_stn 0 --use_rsc 1 --pc_w 0.3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method identity_random --device 7
python train.py --exp_name base8_sysu_baseline_VBS_CHRS_SEM --use_SEM --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method camera_random --device 6
python train.py --exp_name base8_sysu_baseline_RSA_VBS_SEM --use_SEM  --use_stn 0 --use_rsc 1 --pc_w 0.3  --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method camera_random --device 5
python train.py --exp_name base8_sysu_baseline_RSA_pcw5e-1 --use_stn 0 --use_rsc 1 --pc_w 0.5 --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method identity_random --device 0

## 试着调一调参超过saai
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3_bs16_8 --use_stn 0 --cfg configs/SYSU_batch_16_8.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3_bs10_8 --use_stn 0 --cfg configs/SYSU_batch_10_8.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3



# 9.5
python train.py --exp_name base8_sysu_VBS_RSA_pcw5e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.5 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 0
python train.py --exp_name base8_sysu_VBS_RSA_pcw4e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.4 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 2
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3 # 幽默怎么这个最高？
python train.py --exp_name base8_sysu_VBS_RSA_pcw1e-1_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.1 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 7


# 9.6
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_decrease_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --moda_type decrease --device 5
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_average_modawoforward_SEM_CHRS_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --moda_type average --device 0


python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start0_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 0 --group_ce_w 0.001 --group_cs_w 0.001 --device 0
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start80_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 80 --group_ce_w 0.001 --group_cs_w 0.001 --device 2
python train.py --exp_name base8_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start120_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --device 3


python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start120_gpcew1e-3_gpcsw1e-3 --cs_w 1 --sample_method camera_random --pc_w 0.3 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --device 6
python train.py --exp_name base8_regdb_VBS_RSA_pcw5e-1_modawoforward_SEM_CHRS_start120_gpcew1e-3_gpcsw1e-3 --cs_w 1 --sample_method camera_random --pc_w 0.5 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --device 7


# 9.7 regdb调sota
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_modawoforward_SEM_bs10_8 --cs_w 1 --sample_method camera_random --pc_w 0.3 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --group_ce_w 0.001 --group_cs_w 0.001 --device 4
python train.py --exp_name base8_regdb_VBS_RSA_pcw1e-1_modawoforward_SEM_bs10_8 --cs_w 1 --sample_method camera_random --pc_w 0.1 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --group_ce_w 0.001 --group_cs_w 0.001 --device 5

python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_modawoforward_SEM_bs10_8 --cs_w 1 --sample_method camera_random --pc_w 0.3 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --group_ce_w 0.001 --group_cs_w 0.001 --device 4
python train.py --exp_name base8_regdb_VBS_RSA_pcw1e-1_modawoforward_SEM_bs10_8 --cs_w 1 --sample_method camera_random --pc_w 0.1 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --group_ce_w 0.001 --group_cs_w 0.001 --device 5

python train.py --exp_name base8_regdb_VBS_RSA_pcw1e-1_modawoforward_SEM_bs10_10 --cs_w 1 --sample_method camera_random --pc_w 0.1 --use_SEM --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --group_ce_w 0.001 --group_cs_w 0.001  --device 4
python train.py --exp_name base8_regdb_VBS_RSA_pcw1e-1_modawoforward_bs10_10 --cs_w 1 --sample_method camera_random --pc_w 0.1 --cfg models/model_base8/RegDB.yml --extra_start_epoch 114514 --group_ce_w 0.001 --group_cs_w 0.001  --device 5


## 不使用CHSR，batch size设置为10*8，使用SEM，调试pc_w
## 我傻了，之前的regdb实验都用了STN，之前的结果没法看
python train.py --exp_name base8_regdb_VBS_RSA_pcw0_SEM_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0 --use_SEM --device 2
python train.py --exp_name base8_regdb_VBS_RSA_pcw1e-1_SEM_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.1 --use_SEM --device 3
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --device 4
python train.py --exp_name base8_regdb_VBS_RSA_pcw5e-1_SEM_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.5 --use_SEM --device 6
python train.py --exp_name base8_regdb_VBS_RSA_pcw7e-1_SEM_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.7 --use_SEM --device 7

## pcw=0.3最为合适，在这个基础上继续进行调CHSR，先取出这个的权重，接着在这个权重的基础上继续训练
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120continue_CHSR_extra120_gpcew1e-3_gpcsw1e-3 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --start_train_epoch 121 --train_epoch 160 --resume /home/lky/scvd/ckpt/base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120epoch.pth --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --device 7
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120continue_CHSR_extra130_gpcew1e-3_gpcsw1e-3 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --start_train_epoch 121 --train_epoch 160 --resume /home/lky/scvd/ckpt/base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120epoch.pth --extra_start_epoch 130 --group_ce_w 0.001 --group_cs_w 0.001 --device 2
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120continue_CHSR_extra140_gpcew1e-3_gpcsw1e-3 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --start_train_epoch 121 --train_epoch 160 --resume /home/lky/scvd/ckpt/base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120epoch.pth --extra_start_epoch 140 --group_ce_w 0.001 --group_cs_w 0.001 --device 3
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120continue_CHSR_extra150_gpcew1e-3_gpcsw1e-3 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --start_train_epoch 121 --train_epoch 160 --resume /home/lky/scvd/ckpt/base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120epoch.pth --extra_start_epoch 150 --group_ce_w 0.001 --group_cs_w 0.001 --device 4

## 调batch_size
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_CHSR_extra120_gpcew1e-3_gpcsw1e-3_bs10_10 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM   --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --device 0
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_CHSR_extra130_gpcew1e-3_gpcsw1e-3_bs10_10 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM  --extra_start_epoch 130 --group_ce_w 0.001 --group_cs_w 0.001 --device 2

## 调group_loss
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120continue_CHSR_extra130_gpcew1e-2_gpcsw1e-2 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --start_train_epoch 121 --train_epoch 160 --resume /home/lky/scvd/ckpt/base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120epoch.pth --extra_start_epoch 130 --group_ce_w 0.01 --group_cs_w 0.01 --device 3
python train.py --exp_name base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120continue_CHSR_extra130_gpcew1e-1_gpcsw1e-1 --cfg models/model_base8/RegDB.yml --sample_method camera_random --pc_w 0.3 --use_SEM --start_train_epoch 121 --train_epoch 160 --resume /home/lky/scvd/ckpt/base8_regdb_VBS_RSA_pcw3e-1_SEM_bs10_8_120epoch.pth --extra_start_epoch 130 --group_ce_w 0.1 --group_cs_w 0.1 --device 4







python train.py --exp_name base8_sysu_baseline_RSA --use_stn 0 --use_rsc 1 --pc_w 0.3  --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method identity_random --device 7
python train.py --exp_name base8_sysu_baseline_RSA_CHRS --use_stn 0 --use_rsc 1 --pc_w 0.3 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method identity_random --device 6
python train.py --exp_name base8_sysu_baseline_VBS_CHRS --use_stn 0 --use_rsc 0 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method camera_random --device 7





# 9.8 使用rerank，重新跑消融
## 先跑pcw=0.2时的extra_start_epoch
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_SEM_CHRS_start0_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 0 --group_ce_w 0.001 --group_cs_w 0.001 --device 5
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_SEM_CHRS_start80_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 80 --group_ce_w 0.001 --group_cs_w 0.001 --device 4
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_SEM_CHRS_start120_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 120 --group_ce_w 0.001 --group_cs_w 0.001 --device 3
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_SEM_CHRS_start160_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 161 --group_ce_w 0.001 --group_cs_w 0.001 --device 2

# 再跑pcw=0.2, epsilon=40时的消融
python train.py --exp_name base8_sysu_baseline_RSA_pcw2e-1 --use_stn 0 --use_rsc 1 --pc_w 0.2  --cfg configs/SYSU.yml --extra_start_epoch 114514 --sample_method identity_random --device 0
python train.py --exp_name base8_sysu_baseline_RSA_CHRS_pcw2e-1_start40 --use_stn 0 --use_rsc 1 --pc_w 0.2 --cfg configs/SYSU.yml --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --sample_method identity_random --device 1

# baseline太高了，去掉sem和mam来降低baseline
## 改了sysu.yml的modality_attention值，记得改回去
python train.py --exp_name base8_sysu_baseline_woSEM_woMAM --use_stn 0 --use_rsc 0 --cfg configs/SYSU_woMAM.yml --cs_w 1 --extra_start_epoch 114514 --sample_method identity_random --device 0  # 81.47
python train.py --exp_name base8_sysu_baseline_woSEM_woMAM_csw0 --use_stn 0 --use_rsc 0 --cfg configs/SYSU_woMAM.yml --cs_w 0 --extra_start_epoch 114514 --sample_method identity_random --device 1  # 80.65
python train.py --exp_name base8_sysu_baseline_woSEM_woMAM_csw0_bs10_8 --use_stn 0 --use_rsc 0 --cfg configs/SYSU_woMAM.yml --cs_w 0 --extra_start_epoch 114514 --sample_method identity_random --device 2
python train.py --exp_name base8_sysu_baseline_woSEM_woMAM_csw0_bs16_8 --use_stn 0 --use_rsc 0 --cfg configs/SYSU_woMAM.yml --cs_w 0 --extra_start_epoch 114514 --sample_method identity_random --device 3

python train.py --exp_name base8_sysu_baseline_woSEM_woMAM_csw0 --use_stn 0 --use_rsc 0 --cfg configs/SYSU_woMAM.yml --cs_w 0 --extra_start_epoch 114514 --sample_method identity_random --device 1



# 12.5 icme
## 做group的消融实验，我已经在loss里乘了0.001
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew1_gpcsw1 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 6 # the best 现在不是了，改了之后数值不稳定
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew0_gpcsw0 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0 --group_cs_w 0 --device 7
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew05_gpcsw05 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 0.5 --group_cs_w 0.5 --device 0

# 12.6 icme
## 继续做group的消融实验
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew15_gpcsw15 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 1.5 --group_cs_w 1.5 --device 1
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew2_gpcsw2 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 2.0 --group_cs_w 2.0 --device 2
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew25_gpcsw25 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 2.5 --group_cs_w 2.5 --device 6
python train.py --exp_name icme_sysu_VBS_RSA_pcw3e-1_modawoforward_SEM_CHRS_start40_gpcew3_gpcsw3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.3 --use_SEM --extra_start_epoch 40 --group_ce_w 3 --group_cs_w 3 --device 7

# 12.7 草上面的实验又做错了
## 论文里面展示的最好效果是这个
python train.py --exp_name base8_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1e-3_gpcsw1e-3 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0.001 --group_cs_w 0.001 --device 3 # 幽默怎么这个最高？这个是论文里面的结果 r1: 86.26 mAP:83.91
## 做group的消融实验，我已经在loss里乘了0.001
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 1
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew0_gpcsw0 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0 --group_cs_w 0 --device 2
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew05_gpcsw05 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0.5 --group_cs_w 0.5 --device 6
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew15_gpcsw15 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1.5 --group_cs_w 1.5 --device 7

## 继续消融，先接着跑灵敏性，在跑groupid groupcs
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew2_gpcsw2 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 2 --group_cs_w 2 --device 1
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew25_gpcsw25 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 2.5 --group_cs_w 2.5 --device 4

python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw0 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 0 --device 2
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew0_gpcsw1 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0 --group_cs_w 1 --device 3


# 12.10 消融实验
## base+rsa(full)+chsr(wo group loss)
python train.py --exp_name icme_sysu_woVBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew0_gpcsw0 --sample_method identity_random --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 0 --group_cs_w 0 --device 5
## base+rsa(wo l_ma)
python train.py --exp_name icme_sysu_woVBS_RSA_pcw0_modawoforward_SEM_woCHSR --sample_method identity_random --use_stn 0 --cfg configs/SYSU.yml --pc_w 0 --use_SEM --extra_start_epoch 114514 --group_ce_w 0 --group_cs_w 0 --device 7

# 12.11
## 跑rsa模块数的消融，别忘了初始是3*2
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_rsa12 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 1
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_rsa22 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 2
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_rsa42 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 4
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_rsa52 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 5
# 换成5*2效果提升一大截 rerank:
# 2024-12-12 05:43:22,841 all num-shot:1 r1 precision = 87.64 , r10 precision = 98.93 , r20 precision = 99.50, mAP = 85.29


## rsa模型选择
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_lstm --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --rsa_model lstm --device 1
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_transformerEncoder --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --rsa_model transformerEncoder --device 5
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_transformerDecoder --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --rsa_model transformerDecoder --device 6



# 12.22 chsr中的m和\hat{k}的消融
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_m2 --top_k 2 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 0
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_m6 --top_k 6 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 1
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_m8 --top_k 8 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 2

python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_hatk4 --extra_img_num 4 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 3
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_hatk6 --extra_img_num 6 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 4
python train.py --exp_name icme_sysu_VBS_RSA_pcw2e-1_modawoforward_SEM_CHSR_start40_gpcew1_gpcsw1_hatk8 --extra_img_num 8 --use_stn 0 --cfg configs/SYSU.yml --pc_w 0.2 --use_SEM --extra_start_epoch 40 --group_ce_w 1 --group_cs_w 1 --device 5

# 2025.2.20
## RegDB上进行消融实验
### baseline
python train.py --exp_name base8_regdb_bs10_8_baseline --cfg models/model_base8/RegDB.yml --use_rsc 0 --sample_method identity_random --pc_w 0.0 --use_SEM  --extra_start_epoch 114514 --device 0
### baseline+rsa
python train.py --exp_name base8_regdb_RSA_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0 --use_SEM --extra_start_epoch 114514 --device 1
### baseline+rsa+ma_loss
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1 --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 2
### baseline+rsa+ma_loss+chsr
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_chsr --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 120 --device 3
### baseline+rsa+ma_loss+chsr+retrain_loss
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_chsr_gce1_gcs1 --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 120 --group_ce_w 1 --group_cs_w 1 --device 7
### 全部（不跑了）


# 2025.2.22
## RegDB上跑不同的rsa对比时间
### 论文中原版rsa
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_bilstm --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 0
### 使用单向lstm的rsa
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_lstm --rsa_model lstm --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 1
### transformerEncoder
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_encoder --rsa_model transformerEncoder --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 2
### transformerDecoder
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_decoder --rsa_model transformerDecoder --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 3
### gap
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_gap --use_rsc 0 --cfg models/model_base8/RegDB.yml --sample_method identity_random --use_SEM --extra_start_epoch 114514 --device 4

















# 2025.3.3
## RegDB上进行消融实验
### baseline
python train.py --exp_name base8_regdb_bs10_8_baseline --cfg models/model_base8/RegDB.yml --use_rsc 0 --sample_method identity_random --pc_w 0.0 --use_SEM  --extra_start_epoch 114514 --device 0
### baseline+rsa
python train.py --exp_name base8_regdb_RSA_bs10_8 --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0 --use_SEM --extra_start_epoch 114514 --device 1
### baseline+rsa+ma_loss
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1 --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 2
### baseline+rsa+ma_loss+chsr
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_chsr --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 120 --device 3
### baseline+rsa+ma_loss+chsr+retrain_loss
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_chsr_gce1_gcs1 --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 120 --group_ce_w 1 --group_cs_w 1 --device 7
### 全部（不跑了）


# 2025.3.5
## RegDB上跑不同的rsa对比时间
### 论文中原版rsa
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_bilstm --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 0
### 使用单向lstm的rsa
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_lstm --rsa_model lstm --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 1
### transformerEncoder
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_encoder --rsa_model transformerEncoder --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 2
### transformerDecoder
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_decoder --rsa_model transformerDecoder --cfg models/model_base8/RegDB.yml --sample_method identity_random --pc_w 0.2 --use_SEM --extra_start_epoch 114514 --device 3
### gap
python train.py --exp_name base8_regdb_RSA_bs10_8_ma2e-1_gap --use_rsc 0 --cfg models/model_base8/RegDB.yml --sample_method identity_random --use_SEM --extra_start_epoch 114514 --device 4