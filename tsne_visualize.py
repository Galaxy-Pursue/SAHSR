import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import yaml

from data import get_train_loader
from models.baseline import Baseline
import warnings

warnings.filterwarnings("ignore")


def train(cfg):
    cfg.p_size = 8
    cfg.k_size = 8
    model_name = 'sota'
    if model_name == 'baseline':
        ckpt_pth = 'C:/pyProject/vireid/ckpt/icassp/baseline_vbs.pth'
        weights = torch.load(ckpt_pth)
        cfg.use_stn=0
        cfg.use_SEM = 0
        cfg.use_rsc = 0
    else:
        ckpt_pth = 'C:/pyProject/vireid/ckpt/icassp/sota.pth'
        weights = torch.load(ckpt_pth)
        cfg.use_stn = 0
        cfg.use_SEM = 1
        cfg.use_rsc = 1


    # training data loader
    train_loader, train_dataset = get_train_loader(dataset=cfg.dataset,
                                                   root='C:/pyProject/vireid/SYSU-MM01',
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
                     margin1=cfg.margin1,
                     margin2=cfg.margin2,
                     dp=cfg.dp,
                     dp_w=cfg.dp_w,
                     cs_w=cfg.cs_w,
                     inforce_w=cfg.inforce_w,
                     inforce_type=cfg.inforce_type,
                     cs_type=cfg.cs_type,
                     use_ada=cfg.use_ada,
                     up_margin=cfg.up_margin,
                     down_margin=cfg.down_margin,
                     top_k=cfg.top_k,
                     extra_img_num=cfg.extra_img_num,
                     group_inforce_w=cfg.group_inforce_w,
                     group_cs_w=cfg.group_cs_w,
                     group_ce_w=cfg.group_ce_w,
                     pc_w=cfg.pc_w,
                     use_stn=cfg.use_stn,
                     use_rsc=cfg.use_rsc,
                     use_SEM=cfg.use_SEM,
                     LUP_pretrained=cfg.LUP_pretrained,
                     moda_type=cfg.moda_type,
                     )


    # model.load_state_dict(weights)
    model.cuda()

    # 0:[b c h w] 1:labels 2:cam_ids
    batch = next(iter(train_loader))
    imgs = batch[0].cuda()
    labels = batch[1].cuda()
    cam_ids = batch[2].cuda()
    model.eval()
    feats = model(imgs, labels=labels, cam_ids=cam_ids)

    # Step 1: Map labels to range [1, cfg.p_size]
    unique_labels = torch.unique(labels)
    label_map = {label.item(): idx + 1 for idx, label in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_map[label.item()] for label in labels], device='cuda')

    # Step 2: Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    feats_np = feats.detach().cpu().numpy()  # Convert to numpy for t-SNE
    feats_tsne = tsne.fit_transform(feats_np)
    # Set font size and font type globally
    # plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})
    # Step 3: Create plot with academic color palette and camera-specific markers
    plt.figure(figsize=(12, 12))

    # Normalize labels for color mapping
    normalized_labels = mapped_labels.cpu().numpy()

    # Use seaborn academic color palette (e.g., "Set2" or "muted") with a maximum of 8 colors
    palette = np.array(sns.color_palette("Set2", 8))  # Using Set2 with 8 colors

    # Use Times New Roman font and increase font size
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16  # Set font size
    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 这里的2是边框的粗细，可以根据需要调整

    # Step 4: Scatter plot with appropriate markers and increased marker sizes
    for i, (x, y) in enumerate(feats_tsne):
        label_idx = mapped_labels[i].item() - 1  # Get label index for color
        cam_id = cam_ids[i].item()

        # Use triangle for cam_id = 3 or 6, circle otherwise, with larger markers
        marker = '^' if cam_id in [3, 6] else 'o'
        plt.scatter(x, y, color=palette[label_idx % 8], marker=marker, edgecolor='k', s=650)  # Larger marker size

    # Add title with larger font size
    # plt.title('t-SNE Visualization', fontsize=20)

    # Show plot
    plt.show()



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
    parser.add_argument("--ma_w", type=float, default=0.2)
    parser.add_argument("--cs_w", type=float, default=1)
    parser.add_argument("--group_ce_w", type=float, default=1.0)
    parser.add_argument("--group_cs_w", type=float, default=1.0)
    parser.add_argument("--up_margin", type=float, default=1.0)
    parser.add_argument("--down_margin", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--extra_img_num", type=int, default=2)
    parser.add_argument("--extra_start_epoch", type=int, default=40)
    parser.add_argument("--use_rsa", action='store_true')
    parser.add_argument("--sample_method", type=str, default="camera_random")
    parser.add_argument("--moda_type", type=str, default="increase")
    parser.add_argument("--rsa_model", type=str, default="bilstm")

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
    cfg.cs_w = args.cs_w
    ## These are new added parameters
    cfg.up_margin = args.up_margin
    cfg.down_margin = args.down_margin
    cfg.top_k = args.top_k
    cfg.extra_img_num = args.extra_img_num
    cfg.extra_start_epoch = args.extra_start_epoch
    cfg.group_cs_w = args.group_cs_w
    cfg.group_ce_w = args.group_ce_w
    cfg.ma_w = args.ma_w
    cfg.use_rsa = args.use_rsa
    cfg.sample_method = args.sample_method
    cfg.moda_type = args.moda_type
    cfg.rsa_model = args.rsa_model

    cfg.file_prefix = args.file_prefix
    # cfg.resume = args.resume
    cfg.start_train_epoch = args.start_train_epoch
    if cfg.resume: cfg.num_epoch = args.train_epoch
    cfg.freeze()

    train(cfg)