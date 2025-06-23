import torch
import numpy as np

# from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from utils.calc_acc import calc_acc
from torch.nn import functional as F


def create_train_engine(model, optimizer, non_blocking=False, p=12):
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

        # During normal training，we need to calculate centerlogits for extra_train sampling
        if extra_train == False:
            # calculate mean logits of every id, finding the ones with highest confidence
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


        # use group loss for extra_train
        else:
            center_logits = None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return metric, center_logits
    engine = Engine(_process_func)

    @engine.on(Events.STARTED)
    def initialize_state(engine):
        engine.state.process_func = _process_func

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
