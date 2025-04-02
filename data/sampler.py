import numpy as np

import copy

import torch
from torch.utils.data import Sampler, Dataset
from collections import defaultdict


class CrossModalityRandomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.rgb_list = []
        self.ir_list = []
        for i, cam in enumerate(dataset.cam_ids):
            if cam in [3, 6]:
                self.ir_list.append(i)
            else:
                self.rgb_list.append(i)

    def __len__(self):
        return max(len(self.rgb_list), len(self.ir_list)) * 2

    def __iter__(self):
        sample_list = []
        rgb_list = np.random.permutation(self.rgb_list).tolist()
        ir_list = np.random.permutation(self.ir_list).tolist()

        rgb_size = len(self.rgb_list)
        ir_size = len(self.ir_list)
        if rgb_size >= ir_size:
            diff = rgb_size - ir_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                ir_list.extend(np.random.permutation(self.ir_list).tolist())
            ir_list.extend(np.random.choice(self.ir_list, pad_size, replace=False).tolist())
        else:
            diff = ir_size - rgb_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                rgb_list.extend(np.random.permutation(self.rgb_list).tolist())
            rgb_list.extend(np.random.choice(self.rgb_list, pad_size, replace=False).tolist())

        assert len(rgb_list) == len(ir_list)

        half_bs = self.batch_size // 2
        for start in range(0, len(rgb_list), half_bs):
            sample_list.extend(rgb_list[start:start + half_bs])
            sample_list.extend(ir_list[start:start + half_bs])

        return iter(sample_list)


class CrossModalityIdentitySampler(Sampler):
    '''
    首先，对所有的id进行随机排列，然后按照p_size的大小划分批次。
    对于每个批次，首先从RGB图像中随机选择k_size个图像，然后从IR图像中随机选择k_size个图像。最后，返回一个包含所有批次的样本索引的迭代器。
    '''
    def __init__(self, dataset, p_size, k_size):
        self.dataset = dataset
        self.p_size = p_size
        self.k_size = k_size // 2
        self.batch_size = p_size * k_size * 2

        self.id2idx_rgb = defaultdict(list)
        self.id2idx_ir = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            if dataset.cam_ids[i] in [3, 6]:
                self.id2idx_ir[identity].append(i)
            else:
                self.id2idx_rgb[identity].append(i)

    def __len__(self):
        return self.dataset.num_ids * self.k_size * 2

    def __iter__(self):
        sample_list = []

        id_perm = np.random.permutation(self.dataset.num_ids)
        for start in range(0, self.dataset.num_ids, self.p_size):
            selected_ids = id_perm[start:start + self.p_size]

            sample = []
            for identity in selected_ids:  # 每一个id随机采样k//2张rgb
                replace = len(self.id2idx_rgb[identity]) < self.k_size
                s = np.random.choice(self.id2idx_rgb[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

            sample.clear()
            for identity in selected_ids:  # 每一个id随机采样k//2张ir
                replace = len(self.id2idx_ir[identity]) < self.k_size
                s = np.random.choice(self.id2idx_ir[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

        return iter(sample_list)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(list)  # 存储每个id的所有rgb图的索引
        self.index_dic_I = defaultdict(list)  # 存储每个id的所有红外图的索引
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_R[identity].append(i)
        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_I[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_R = copy.deepcopy(self.index_dic_R[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_R):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)
            if len(idxs_R) > len(idxs_I):
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I, idxs_R):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class NormTripletSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.data_source.ids):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:  # 为所有的pid，每个准备k张可选的图片
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:  # 当为一个id采样了k张图片时，停止采样，并将采样结果放到batch_idxs_dict
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)  # 从所有可选的pid中无放回采样p个人
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomCameraSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(dict)  # 存储每个id的所有cam，内层字典记录对应的索引
        self.index_dic_I = defaultdict(dict)
        for i, identity in enumerate(data_source.ids):
            cam_id = data_source.cam_ids[i]
            if cam_id in [3, 6]:
                if cam_id in self.index_dic_I[identity]:
                    self.index_dic_I[identity][cam_id].append(i)
                else:
                    self.index_dic_I[identity][cam_id] = [i]
            else:
                if cam_id in self.index_dic_R[identity]:
                    self.index_dic_R[identity][cam_id].append(i)
                else:
                    self.index_dic_R[identity][cam_id] = [i]

        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            num = 99999
            for cam_id in self.index_dic_I[pid].keys():
                num = min(num, len(self.index_dic_I[pid][cam_id]))
            total_num = num * len(self.index_dic_I[pid].keys())
            if total_num < self.num_instances:
                total_num = self.num_instances
            self.length += total_num - total_num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = []
            idxs_R = []
            cam_idxs_I = copy.deepcopy(self.index_dic_I[pid])
            cam_idxs_R = copy.deepcopy(self.index_dic_R[pid])
            cams_I = cam_idxs_I.keys()
            cams_R = cam_idxs_R.keys()
            # 各个rgb和ir的cam都弄得一样长
            min_num_I = 999999
            min_num_R = 999999
            for cam_I in cams_I:
                idxs_I.append(cam_idxs_I[cam_I])
                min_num_I = min(len(cam_idxs_I[cam_I]), min_num_I)
            for cam_R in cams_R:
                idxs_R.append(cam_idxs_R[cam_R])
                min_num_R = min(len(cam_idxs_R[cam_R]), min_num_R)
            # 每个cam的图片数采样到最少图片数的cam
            idxs_I_new = []
            idxs_R_new = []
            for i in range(len(idxs_I)):
                idxs_I_new.extend(np.random.choice(idxs_I[i], size=min_num_I, replace=False))
            for i in range(len(idxs_R)):
                idxs_R_new.extend(np.random.choice(idxs_R[i], size=min_num_R, replace=False))

            if len(idxs_I_new) < self.num_instances // 2 and len(idxs_R_new) < self.num_instances // 2:
                idxs_I_new = np.random.choice(idxs_I_new, size=self.num_instances // 2, replace=True)
                idxs_R_new = np.random.choice(idxs_R_new, size=self.num_instances // 2, replace=True)
            if len(idxs_I_new) > len(idxs_R_new):
                idxs_I_new = np.random.choice(idxs_I_new, size=len(idxs_R_new), replace=False)
            if len(idxs_R_new) > len(idxs_I_new):
                idxs_R_new = np.random.choice(idxs_R_new, size=len(idxs_I_new), replace=False)
            np.random.shuffle(idxs_I_new)
            np.random.shuffle(idxs_R_new)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I_new, idxs_R_new):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomCameraSampler2(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(dict)  # 存储每个id的所有cam，内层字典记录对应的索引
        self.index_dic_I = defaultdict(dict)
        for i, identity in enumerate(data_source.ids):
            cam_id = data_source.cam_ids[i]
            if cam_id in [3, 6]:
                if cam_id in self.index_dic_I[identity]:
                    self.index_dic_I[identity][cam_id].append(i)
                else:
                    self.index_dic_I[identity][cam_id] = [i]
            else:
                if cam_id in self.index_dic_R[identity]:
                    self.index_dic_R[identity][cam_id].append(i)
                else:
                    self.index_dic_R[identity][cam_id] = [i]

        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            num = 0
            for cam_id in self.index_dic_I[pid].keys():
                num = max(num, len(self.index_dic_I[pid][cam_id]))
            total_num = num * len(self.index_dic_I[pid].keys())
            if total_num < self.num_instances:
                total_num = self.num_instances
            self.length += total_num - total_num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = []
            idxs_R = []
            cam_idxs_I = copy.deepcopy(self.index_dic_I[pid])
            cam_idxs_R = copy.deepcopy(self.index_dic_R[pid])
            cams_I = cam_idxs_I.keys()
            cams_R = cam_idxs_R.keys()
            # 各个rgb和ir的cam都弄得和最多的cam一样长
            max_num_I = 0
            max_num_R = 0
            for cam_I in cams_I:
                idxs_I.append(cam_idxs_I[cam_I])
                max_num_I = max(len(cam_idxs_I[cam_I]), max_num_I)
            for cam_R in cams_R:
                idxs_R.append(cam_idxs_R[cam_R])
                max_num_R = max(len(cam_idxs_R[cam_R]), max_num_R)
            # 每个cam的图片数重复采样到最多图片数的cam
            idxs_I_new = []
            idxs_R_new = []
            for i in range(len(idxs_I)):
                idxs_I_new.extend(np.random.choice(idxs_I[i], size=max_num_I, replace=True))  # 有没有必要最多的就不用重复采样
            for i in range(len(idxs_R)):
                idxs_R_new.extend(np.random.choice(idxs_R[i], size=max_num_R, replace=True))

            if len(idxs_I_new) < self.num_instances // 2 and len(idxs_R_new) < self.num_instances // 2:
                idxs_I_new = np.random.choice(idxs_I_new, size=self.num_instances // 2, replace=True)
                idxs_R_new = np.random.choice(idxs_R_new, size=self.num_instances // 2, replace=True)
            if len(idxs_I_new) > len(idxs_R_new):
                idxs_I_new = np.random.choice(idxs_I_new, size=len(idxs_R_new), replace=False)
            if len(idxs_R_new) > len(idxs_I_new):
                idxs_R_new = np.random.choice(idxs_R_new, size=len(idxs_I_new), replace=False)
            np.random.shuffle(idxs_I_new)
            np.random.shuffle(idxs_R_new)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I_new, idxs_R_new):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# WeightedRandomSampler试试这个东西
class extra_sampler():
    '''
    给定待采样的id，对每个id采样k2//2张rgb，k2//2张ir
    '''
    def __init__(self, data_source):
        self.data_source = data_source
        self.index_dic_R = defaultdict(list)  # 存储每个id的所有rgb图的索引
        self.index_dic_I = defaultdict(list)  # 存储每个id的所有红外图的索引
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_R[identity].append(i)


    def __call__(self, id_matrix, img_num=2):
        '''
        id_matrix: 2d ndarray, 用于存储每个id对应分类结果评分最高的id，对这些id进行取样 [[1,2,3],[4,5,6]]
        img_num:对每个id采样几张图像，其中ir和rgb对半分
        '''
        id_list = id_matrix.flatten()
        datas = []
        labels = []
        cam_ids = []
        img_paths = []
        img_ids = []
        for id in id_list:
            idxs_ir = np.random.choice(self.index_dic_I[id], size=img_num//2, replace=False)  # id对应的图像索引
            idxs_rgb = np.random.choice(self.index_dic_R[id], size=img_num//2, replace=False)
            for item in idxs_ir:
                data, label, cam_id, img_path, img_id = self.data_source[item]
                datas.append(data)
                labels.append(label)
                cam_ids.append(cam_id)
                img_ids.append(img_id)
                img_paths.append(img_path)
            for item in idxs_rgb:
                data, label, cam_id, img_path, img_id = self.data_source[item]
                datas.append(data)
                labels.append(label)
                cam_ids.append(cam_id)
                img_ids.append(img_id)
                img_paths.append(img_path)
        datas = torch.stack(datas, dim=0)
        labels = torch.stack(labels, dim=0)
        cam_ids = torch.stack(cam_ids, dim=0)
        return datas, labels, cam_ids, img_paths, img_ids


class DynamicSampler(Sampler):
    def __init__(self, index_dic_R, index_dic_I, img_num=4):
        self.index_dic_R = index_dic_R
        self.index_dic_I = index_dic_I
        self.img_num = img_num
        self.id_matrix = None

    def set_id_matrix(self, id_matrix):
        # self.id_matrix = id_matrix.flatten()
        self.id_matrix = id_matrix.reshape(-1)

    def set_img_num(self, img_num):
        self.img_num = img_num

    def __iter__(self):
        if self.id_matrix is None:
            raise ValueError("id_matrix has not been set.")

        indices = []
        for id in self.id_matrix:
            if len(self.index_dic_I[id]) >= self.img_num // 2 and len(self.index_dic_R[id]) >= self.img_num // 2:
                idxs_ir = np.random.choice(self.index_dic_I[id], size=self.img_num // 2, replace=False)
                idxs_rgb = np.random.choice(self.index_dic_R[id], size=self.img_num // 2, replace=False)
                indices.extend(idxs_ir)
                indices.extend(idxs_rgb)
            else:
                print(f"Warning: Not enough images for id {id}")
        return iter(indices)

    def __len__(self):
        if self.id_matrix is None:
            return 0
        return len(self.id_matrix) * self.img_num

