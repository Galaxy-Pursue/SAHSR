import logging 
import os
import pprint

import torch
import yaml
from torch import optim

from data import get_test_loader, get_train_loader

from engine import get_trainer
from models.baseline import Baseline
from torch.utils.tensorboard import SummaryWriter

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


def train(cfg):
    # set logger
    log_dir = os.path.join(cfg.file_prefix, cfg.prefix, cfg.exp_name, "logs/")
    # set tensorboard
    writer = SummaryWriter(log_dir = os.path.join('./', f'vis_logs_now/{cfg.prefix}/{cfg.exp_name}'))  # logs为保存日志的目录
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + cfg.exp_name,
                        filemode="w")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(pprint.pformat(cfg))
    
    # training data loader
    train_loader, train_dataset = get_train_loader(dataset=cfg.dataset,
                                    root=cfg.data_root,
                                    sample_method=cfg.sample_method,
                                    batch_size=cfg.batch_size,
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    image_size=cfg.image_size,
                                    num_workers=8)
    
    # evaluation data loader
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=64,
                                                       image_size=cfg.image_size,
                                                       num_workers=4)
    # model
    model = Baseline(num_classes=cfg.num_id,
                     backbone=cfg.backbone,
                     pattern_attention=cfg.pattern_attention,
                     modality_attention=cfg.modality_attention,
                     mutual_learning=cfg.mutual_learning,
                     drop_last_stride=cfg.drop_last_stride,
                     triplet=cfg.triplet,
                     p_size=cfg.p_size,
                     k_size=cfg.k_size,
                     center_cluster=cfg.center_cluster,
                     center=cfg.center,
                     margin=cfg.margin,
                     num_parts=cfg.num_parts,
                     weight_KL=cfg.weight_KL,
                     weight_sid=cfg.weight_sid,
                     weight_sep=cfg.weight_sep,
                     update_rate=cfg.update_rate,
                     classification=cfg.classification,
                     margin1 = cfg.margin1,
                     margin2 = cfg.margin2,
                     dp = cfg.dp,
                     dp_w = cfg.dp_w,
                     cs_w = cfg.cs_w,
                     inforce_w= cfg.inforce_w,
                     inforce_type=cfg.inforce_type,
                     cs_type=cfg.cs_type,
                     up_margin=cfg.up_margin,
                     down_margin=cfg.down_margin,
                     top_k=cfg.top_k,
                     extra_img_num=cfg.extra_img_num,
                     group_inforce_w=cfg.group_inforce_w,
                     group_cs_w=cfg.group_cs_w,
                     group_ce_w=cfg.group_ce_w,
                     pc_w=cfg.pc_w,
                     use_rsc=cfg.use_rsc,
                     moda_type=cfg.moda_type,
                     rsa_model=cfg.rsa_model
                     )
    
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    print(get_parameter_number(model))
    
    model.cuda()
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.optimizer_extra == 'adam':
        optimizer_extra = optim.Adam(model.parameters(), lr=cfg.lr_extra, weight_decay=cfg.wd)
    print(f'lr_extra:{cfg.lr_extra}')

    def step_lr_with_warmup(epoch):
        if epoch < 10:
            return (epoch + 1) / 10 
        else:
            if epoch < cfg.lr_step[0]:
                return 1
            elif epoch < cfg.lr_step[1]:
                return 0.1
            elif epoch < 180:
                return 0.01
            elif epoch < 280:
                return 0.005
            elif epoch < 320:
                return 0.002
            else:
                return 0.001
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = cfg.lr
    for param_group in optimizer_extra.param_groups:
        param_group["initial_lr"] = cfg.lr_extra

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=step_lr_with_warmup,last_epoch=cfg.start_train_epoch-1)
    lr_scheduler_extra = optim.lr_scheduler.LambdaLR(optimizer=optimizer_extra, lr_lambda=step_lr_with_warmup,
                                               last_epoch=cfg.start_train_epoch - 1)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint, strict=False)

    # engine
    checkpoint_dir = os.path.join(cfg.file_prefix, cfg.prefix, cfg.exp_name, "checkpoints/")
    engine = get_trainer(dataset=cfg.dataset,
                         model=model,
                         writer=writer,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         optimizer_extra=optimizer_extra,
                         lr_scheduler_extra=lr_scheduler_extra,
                         logger=logger,
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_dir=checkpoint_dir,
                         prefix=cfg.prefix,
                         eval_interval=cfg.eval_interval,
                         start_eval=cfg.start_eval,
                         gallery_loader=gallery_loader,
                         query_loader=query_loader,
                         start_train_epoch = cfg.start_train_epoch,
                         train_dataset=train_dataset,
                         top_k=cfg.top_k,
                         extra_img_num=cfg.extra_img_num,
                         extra_start_epoch=cfg.extra_start_epoch,
                         p_size=cfg.p_size,
                         k_size=cfg.k_size,
                         )

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)
    
if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg
    print('start training')

    parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", type=str, default="models/model_base8/RegDB.yml")
    parser.add_argument("--cfg", type=str, default="configs/SYSU.yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="sysu_test")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--update_rate", type=float, default=0.02)
    parser.add_argument("--num_parts", type=int, default=7)
    parser.add_argument("--margin1", type=float, default=0.01)
    parser.add_argument("--margin2", type=float, default=0.7)
    parser.add_argument("--dp", type=str, default="l2")
    # dp_w: 0.3
    parser.add_argument("--dp_w", type=float, default=0.5)
    parser.add_argument("--pc_w", type=float, default=0.0)
    parser.add_argument("--cs_w", type=float, default=1)
    parser.add_argument("--inforce_w", type=float, default=0.0)
    parser.add_argument("--inforce_type", type=str, default='original')
    parser.add_argument("--group_ce_w", type=float, default=0.0)
    parser.add_argument("--group_inforce_w", type=float, default=0.0)
    parser.add_argument("--group_cs_w", type=float, default=0.0)
    parser.add_argument("--cs_type", type=str, default='original')
    parser.add_argument("--up_margin", type=float, default=1.0)
    parser.add_argument("--down_margin", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--extra_img_num", type=int, default=2)
    parser.add_argument("--extra_start_epoch", type=int, default=40)
    parser.add_argument("--lr_extra", type=float, default=0.00005)
    parser.add_argument("--use_rsc", type=int, default=1)
    parser.add_argument("--sample_method", type=str, default="camera_random")
    parser.add_argument("--moda_type", type=str, default="increase")
    parser.add_argument("--rsa_model", type=str, default="bilstm")
    # 一个很坑的东西，向parser中传bool类型，当type为bool时不能用True或False
    # 在Python中，bool()函数会将非空字符串转换为True。在你的命令行参数中，--use_ada False被解析为字符串"False"，然后bool("False")返回True，这就是为什么args.use_ada的值为True。
    # 下面两个是错误的传参方法
    # parser.add_argument("--cs_projection", type=bool, default=False)
    # parser.add_argument("--use_ada", type=bool, default=False)

    # 要向parser中传入true或false，需要使用action='store_true'或action='store_false'，且命令行的输入也要做相应的修改
    # 如果命令行中包含--use_ada，那么args.use_ada就会被设置为True，否则为False。

    parser.add_argument("--cs_projection", action='store_true')

    parser.add_argument("--file_prefix", type=str, default="./logs/")
    # for continue training
    parser.add_argument("--resume")
    parser.add_argument("--start_train_epoch", type=int, default=0)
    parser.add_argument("--train_epoch", type=int, default=0)
    args = parser.parse_args()

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # enable cudnn backend
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    if cfg.sample_method == 'identity_uniform':
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.exp_name = args.exp_name
    cfg.backbone = args.backbone
    cfg.update_rate = args.update_rate
    cfg.num_parts = args.num_parts
    cfg.margin1 = args.margin1
    cfg.margin2 = args.margin2
    cfg.dp = args.dp
    cfg.dp_w = args.dp_w
    cfg.cs_w = args.cs_w
    ## 下面是新加入的参数
    cfg.inforce_w = args.inforce_w
    cfg.inforce_type = args.inforce_type
    cfg.cs_type = args.cs_type
    cfg.up_margin = args.up_margin
    cfg.down_margin = args.down_margin
    cfg.top_k = args.top_k
    cfg.extra_img_num = args.extra_img_num
    cfg.extra_start_epoch = args.extra_start_epoch
    cfg.lr_extra = args.lr_extra
    cfg.group_inforce_w = args.group_inforce_w
    cfg.group_cs_w = args.group_cs_w
    cfg.group_ce_w = args.group_ce_w
    cfg.pc_w = args.pc_w
    cfg.use_rsc = args.use_rsc
    cfg.sample_method = args.sample_method
    cfg.moda_type = args.moda_type
    cfg.rsa_model = args.rsa_model

    cfg.file_prefix = args.file_prefix
    # cfg.resume = args.resume
    cfg.start_train_epoch = args.start_train_epoch
    if cfg.resume: cfg.num_epoch = args.train_epoch
    cfg.freeze()

    train(cfg)