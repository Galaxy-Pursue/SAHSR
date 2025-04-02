import logging
import os
import numpy as np
import torch
import scipy.io as sio

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from torch.utils.data import DataLoader

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_sysu import eval_sysu, eval_sysu2
from utils.eval_regdb import eval_regdb, eval_regdb2
from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg
from data.sampler import extra_sampler, DynamicSampler
from data.dataset import DynamicSamplerDataset

def get_trainer(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,
                save_dir="checkpoints", prefix="model", gallery_loader=None, query_loader=None,
                eval_interval=None, start_eval=None, start_train_epoch=0, train_dataset=None, top_k=6, extra_img_num=2,extra_start_epoch=0,
                p_size=12, k_size=12, optimizer_extra=None, lr_scheduler_extra=None):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)
        
    # trainer
    trainer = create_train_engine(model, optimizer, optimizer_extra, non_blocking, p=p_size, top_k=top_k, extra_img_num=extra_img_num)

    dynamic_dataset = DynamicSamplerDataset(train_dataset)
    extra_train_sampler = DynamicSampler(dynamic_dataset.index_dic_R, dynamic_dataset.index_dic_I, img_num=extra_img_num)

    
    # checkpoint handler
    handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=2, create_dir=True,
                              save_as_state_dict=True, require_empty=False, start_train_epoch=start_train_epoch)
    # handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=2, create_dir=True,
    #                           save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)

    kv_metric = AutoKVMetric()
    
    # evaluators

    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        inf_evaluator = create_eval_engine(model, 'inf', non_blocking)
        vis_evaluator = create_eval_engine(model, 'vis', non_blocking)
    
    @trainer.on(Events.STARTED)
    def train_start(engine):
        engine.state.epoch = start_train_epoch
        setattr(engine.state, "best_rank1", 0.0)
        setattr(engine.state, "best_rank1_epoch", 0)

    @trainer.on(Events.COMPLETED)
    def train_completed(engine):
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load("{}/model_best.pth".format(save_dir)))
        logging.info(f'Loaded model checkpoint from {save_dir}/model_best.pth || epoch : {engine.state.best_rank1_epoch}')
        # extract query feature
        inf_evaluator.run(query_loader)

        q_feats = torch.cat(inf_evaluator.state.feat_list, dim=0)
        q_ids = torch.cat(inf_evaluator.state.id_list, dim=0).numpy()
        q_cams = torch.cat(inf_evaluator.state.cam_list, dim=0).numpy()
        q_img_paths = np.concatenate(inf_evaluator.state.img_path_list, axis=0)

        # extract gallery feature
        vis_evaluator.run(gallery_loader)

        g_feats = torch.cat(vis_evaluator.state.feat_list, dim=0)
        g_ids = torch.cat(vis_evaluator.state.id_list, dim=0).numpy()
        g_cams = torch.cat(vis_evaluator.state.cam_list, dim=0).numpy()
        g_img_paths = np.concatenate(vis_evaluator.state.img_path_list, axis=0)

        # print("best rank1={:.2f}%".format(engine.state.best_rank1))

        if dataset == 'sysu':
            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            logging.info('no aim:')
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=False)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, aim=False)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, aim=False)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, aim=False)
            logging.info('aim:')
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=True, k1=4, k2=1)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, aim=True, k1=20, k2=6)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, aim=True, k1=2, k2=2)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, aim=True, k1=20, k2=6)
            # logging.info('no rerank:')
            # eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=False)
            # eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=False)
            # eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=False)
            # eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=False)
            logging.info('rerank:')
            eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=True)
            eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=True)
            eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=True)
            eval_sysu2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=True)
        elif dataset == 'regdb':
            logging.info('no aim:')
            logging.info('infrared to visible')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=False)
            logging.info('visible to infrared')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, aim=False)
            logging.info('aim:')
            logging.info('infrared to visible aim')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=True, k1=8, k2=2)
            logging.info('visible to infrared aim')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, aim=True, k1=8, k2=2)
            logging.info('no rerank')
            print('infrared to visible')
            eval_regdb2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=False)
            print('visible to infrared')
            eval_regdb2(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=False)
            logging.info('rerank')
            print('infrared to visible')
            eval_regdb2(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=True)
            print('visible to infrared')
            eval_regdb2(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=True)
        elif dataset == 'market':
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=engine.aim)
        # elif dataset == 'llcm':
        #     logging.info('no rerank:')
        #     logging.info('infrared to visible')
        #     eval_llcm(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=False)
        #     logging.info('visible to infrared')
        #     eval_llcm(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=False)
        #     logging.info('rerank:')
        #     logging.info('infrared to visible')
        #     eval_llcm(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=True)
        #     logging.info('visible to infrared')
        #     eval_llcm(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=True)

        inf_evaluator.state.feat_list.clear()
        inf_evaluator.state.id_list.clear()
        inf_evaluator.state.cam_list.clear()
        inf_evaluator.state.img_path_list.clear()
        vis_evaluator.state.feat_list.clear()
        vis_evaluator.state.id_list.clear()
        vis_evaluator.state.cam_list.clear()
        vis_evaluator.state.img_path_list.clear()
        del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

        torch.cuda.empty_cache()
        
    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
    
        epoch = engine.state.epoch
        if model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_

        kv_metric.reset()
        timer.reset()
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch
        if epoch == 120:
            torch.save(model.state_dict(), "{}/model_120epoch.pth".format(save_dir))

        if lr_scheduler is not None:
            lr_scheduler.step()
        if lr_scheduler_extra is not None:
            lr_scheduler_extra.step()
            
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('lr_extra', optimizer_extra.param_groups[0]['lr'], epoch)

        if epoch % eval_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if inf_evaluator and vis_evaluator and epoch % eval_interval == 0 and epoch > start_eval:
            torch.cuda.empty_cache()

            # extract query feature
            inf_evaluator.run(query_loader)

            q_feats = torch.cat(inf_evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(inf_evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(inf_evaluator.state.cam_list, dim=0).numpy()
            q_img_paths = np.concatenate(inf_evaluator.state.img_path_list, axis=0)

            # extract gallery feature
            vis_evaluator.run(gallery_loader)

            g_feats = torch.cat(vis_evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(vis_evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(vis_evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(vis_evaluator.state.img_path_list, axis=0)

            if dataset == 'sysu':
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                mAP, r1, r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, aim=True, k1=4, k2=1)#, aim=engine.aim)
            elif dataset == 'regdb':
                print('infrared to visible')
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=False)
                print('visible to infrared')
                mAP, r1_, r5, _, _ = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, aim=False)
                r1 = (r1 + r1_) / 2
            elif dataset == 'market':
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, aim=engine.aim)
            
            if r1 > engine.state.best_rank1:
                engine.state.best_rank1 = r1
                engine.state.best_rank1_epoch = epoch
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))

            if writer is not None:
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)

            inf_evaluator.state.feat_list.clear()
            inf_evaluator.state.id_list.clear()
            inf_evaluator.state.cam_list.clear()
            inf_evaluator.state.img_path_list.clear()
            vis_evaluator.state.feat_list.clear()
            vis_evaluator.state.id_list.clear()
            vis_evaluator.state.cam_list.clear()
            vis_evaluator.state.img_path_list.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

            torch.cuda.empty_cache()
            
    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()

        # print(engine.state.output)
        kv_metric.update(engine.state.output[0])

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - start_train_epoch - 1) * len(engine.state.dataloader)

        if epoch >= extra_start_epoch:
            ## 加入额外训练
            # 这个labels是原始的sampler输出的label，同一个id的所有k张图片在一起，ir和rgb交替着，和训练用的分开不一样（那个操作在engine的_process_func中）
            labels = engine.state.batch[1]
            center_logits = engine.state.output[1]

            # 找到每一行最大的top_k个的索引用来进行额外的训练
            _, topk_indices = torch.topk(center_logits, top_k, dim=1, largest=True)

            # 在新的训练batch中加入自身。如果自己的评分不在最高的top_k个中，则把评分最低的从中剔除换上自己
            for i in range(0, p_size*k_size, k_size):
                identity = labels[i]
                if identity in topk_indices[i // k_size]:
                    pass
                else:
                    topk_indices[i // k_size][-1] = identity


            # 将结果存储在一个二维的 ndarray 中
            topk_indices = topk_indices.cpu().numpy()

            # 设置新的 id_matrix 和 img_num
            extra_train_sampler.set_id_matrix(topk_indices)
            extra_train_sampler.set_img_num(extra_img_num)
            extra_train_loader = DataLoader(dynamic_dataset, batch_size=p_size * top_k * extra_img_num,
                                            sampler=extra_train_sampler, num_workers=16)

            batch = next(iter(extra_train_loader))
            # 不算center，减少计算量
            engine.state.process_func(engine, batch, extra_train=True)
            # engine.state.normal_iter = True


        if iter_in_epoch % log_period == 0 and iter_in_epoch > 0:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()

            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)

            metric_dict = kv_metric.compute()

            # log output information
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "\t%s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)

                logger.info(msg)

            kv_metric.reset()
            timer.reset()

    return trainer