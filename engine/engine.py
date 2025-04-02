import torch
import numpy as np

# from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from utils.calc_acc import calc_acc
from torch.nn import functional as F


def create_train_engine(model, optimizer, optimizer_extra=None, non_blocking=False, p=12, top_k=4, extra_img_num=2):
    device = torch.device("cuda", torch.cuda.current_device())
    scaler = torch.cuda.amp.GradScaler()
    # scaler_extra = torch.cuda.amp.GradScaler()

    def _process_func(engine, batch, extra_train=False):
        model.train()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch

        sub = (cam_ids == 3) + (cam_ids == 6)
        data_inf = data[sub,:,:,:]
        data_vis = data[~sub,:,:,:]
        labels_inf = labels[sub]
        labels_vis = labels[~sub]
        cam_ids_inf = cam_ids[sub]
        cam_ids_vis = cam_ids[~sub]
        data = torch.cat((data_inf, data_vis), dim=0)
        labels = torch.cat((labels_inf, labels_vis), dim=0)
        cam_ids = torch.cat((cam_ids_inf, cam_ids_vis), dim=0)
        
        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        with torch.autocast('cuda', torch.float16):
            loss, metric, logits = model(data, labels,
                                    cam_ids=cam_ids,
                                    epoch=epoch, extra_train=extra_train)

        # 在常规训练时，需要计算centerlogits以备extra_train采样使用
        if extra_train == False:
            # 算一下每个id对应的平均logits，找到平均置信度最高的拿出去进行额外训练
            n = logits.size()[0]
            k = n // p
            # Come to centers
            centers = []
            for i in range(0, n // 2, k // 2):
                centers.append(logits[labels == labels[i]].mean(0))
            center_logits = torch.stack(centers)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        # 当在extra_train时，采用一个新的loss
        else:
            center_logits = None
            # # 在extra_train中，batch一共被分为p组，每组top_k个人，每个人extra_img_num张图像
            # # 前面半个batch是p组，每组top_k个人，每个人extra_img_num张ir图；
            # # 后面半个batch是p组，每组top_k个人，每个人extra_img_num张rgb图；
            # n = logits.size()[0]
            # loss_extra = torch.tensor(0.0, device=device, requires_grad=False)
            # img_num = top_k*extra_img_num//2
            #
            # # 切分批次数据
            # ir_logits = logits[:n // 2].reshape(p, img_num, -1)
            # rgb_logits = logits[n // 2:].reshape(p, img_num, -1)
            # ir_labels = labels[:n // 2].reshape(p, img_num)
            # rgb_labels = labels[n // 2:].reshape(p, img_num)
            # # logits:[p, img_num, c]  labels:[p, img_num]
            # logits = torch.cat([ir_logits, rgb_logits], dim=1)
            # labels = torch.cat([ir_labels, rgb_labels], dim=1)
            #
            # for person_id in range(p):
            #     group_logits = logits[person_id]
            #     group_labels = labels[person_id]
            #
            #     # 找到 group_labels 中的 k 个不同类别
            #     unique_group_labels = torch.unique(group_labels)
            #
            #     # 创建映射字典，将原始类别映射到 0~k-1 的范围内
            #     label_to_index = {lab.item(): idx for idx, lab in enumerate(unique_group_labels)}
            #
            #     # 创建新的 labels 向量
            #     group_labels_new = torch.tensor([label_to_index[lab.item()] for lab in group_labels], dtype=torch.long, device=device)
            #
            #     # 通过高级索引提取 logits 中对应的元素
            #     indices = unique_group_labels.unsqueeze(0).expand(group_logits.size(0), -1)
            #     group_logits_new = torch.gather(group_logits, 1, indices)
            #
            #     # 计算交叉熵损失
            #     loss_extra += F.cross_entropy(group_logits_new, group_labels_new)
            # loss_extra = F.cross_entropy(logits.float(), labels)

            # scaler_extra.scale(loss).backward()
            # scaler_extra.step(optimizer_extra)
            # scaler_extra.update()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return metric, center_logits
    engine = Engine(_process_func)

    @engine.on(Events.STARTED)
    def initialize_state(engine):
        engine.state.process_func = _process_func
        # 记录engine是在算常规训练还是额外训练，从而加快训练速度
        # engine.state.normal_iter = True

    return engine


def create_eval_engine(model, mode, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            feat = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking), mode = mode)

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
