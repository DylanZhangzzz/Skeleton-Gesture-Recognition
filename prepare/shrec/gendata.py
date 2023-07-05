import pickle
from tqdm import tqdm
import sys
from rotation import *
from normalize_skeletons import normalize_skeletons
from prepare import aug_tools
sys.path.extend(['../../'])

import numpy as np
import os
from random import randint, shuffle


def data_aug(skeleton, ag_id):
    def scale(skeleton):
        ratio = 0.2
        low = 1 - ratio
        high = 1 + ratio
        factor = np.random.uniform(low, high)
        video_len = skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(22):
                skeleton[t][j_id] *= factor
        skeleton = np.array(skeleton)
        return skeleton

    def shift(skeleton):
        low = -0.1
        high = -low
        offset = np.random.uniform(low, high, 3)
        video_len = skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(22):
                skeleton[t][j_id] += offset
        skeleton = np.array(skeleton)
        return skeleton

    def noise(skeleton):
        low = -0.1
        high = -low
        # select 4 joints
        all_joint = list(range(22))
        shuffle(all_joint)
        selected_joint = all_joint[0:4]
        time_len = skeleton.shape[0]

        for j_id in selected_joint:
            noise_offset = np.random.uniform(low, high, 3)
            for t in range(time_len):
                skeleton[t][j_id] += noise_offset
        skeleton = np.array(skeleton)
        return skeleton

    def time_interpolate(skeleton):
        skeleton = np.array(skeleton)
        video_len = skeleton.shape[0]

        r = np.random.uniform(0, 1)

        result = []

        for i in range(1, video_len):
            displace = skeleton[i] - skeleton[i - 1]  # d_t = s_t+1 - s_t
            displace *= r
            result.append(skeleton[i - 1] + displace)  # r*disp

        while len(result) < video_len:
            result.append(result[-1])  # padding
        result = np.array(result)
        return result

    # og_id = np.random.randint(3)
    # aug_num = 4
    # ag_id = randint(0, aug_num - 1)
    if ag_id == 0:
        skeleton = scale(skeleton)
    elif ag_id == 1:
        skeleton = shift(skeleton)
    elif ag_id == 2:
        skeleton = noise(skeleton)
    elif ag_id == 3:
        skeleton = time_interpolate(skeleton)

    skeleton = np.expand_dims(np.array(skeleton).transpose((2, 0, 1)), axis=-1)  # CTVM
    return skeleton


def read_skeleton(ske_txt):
    ske_txt = open(ske_txt, 'r').readlines()
    skeletons = []
    for line in ske_txt:
        nums = line.split(' ')
        # num_frame = int(nums[0]) + 1
        coords_frame = np.array(nums).reshape((22, 3)).astype(np.float32)
        skeletons.append(coords_frame)
    num_frame = len(skeletons)
    skeletons = np.expand_dims(np.array(skeletons).transpose((2, 0, 1)), axis=-1)  # CTVM
    skeletons = np.transpose(skeletons, [3, 1, 2, 0])  # M, T, V, C
    # print(skeletons.shape)
    return skeletons, num_frame


def gendata(aug=True):
    root = 'C:/ML/dataset/HandGestureDataset_SHREC2017/'
    train_split = open(os.path.join(root, 'train_gestures.txt'), 'r').readlines()
    val_split = open(os.path.join(root, 'test_gestures.txt'), 'r').readlines()

    skeletons_all_train = []
    names_all_train = []
    labels14_all_train = []
    labels28_all_train = []
    skeletons_all_val = []
    names_all_val = []
    labels14_all_val = []
    labels28_all_val = []

    for line in tqdm(train_split):
        line = line.rstrip()
        g_id, f_id, sub_id, e_id, label_14, label_28, size_seq = map(int, line.split(" "))
        src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt"
                                .format(g_id, f_id, sub_id, e_id))
        skeletons, num_frame = read_skeleton(src_path)
        # print(skeletons[:, 0, 1, :])
        # print(' ', skeletons.shape)
        skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 10])
        # ske_vis(skeletons, view=1, pause=0.1)
        # print('==', skeletons.shape)
        skeletons_all_train.append(skeletons)
        labels14_all_train.append(label_14-1)
        labels28_all_train.append(label_28-1)
        names_all_train.append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))
        if aug is True:
            # print(skeletons.shape)
            p = np.squeeze(skeletons).transpose(1, 2, 0)
            # print(p.shape)
            data = data_aug(p, 0)
            # print(data.shape)
            skeletons_all_train.append(data)
            labels14_all_train.append(label_14 - 1)
            labels28_all_train.append(label_28 - 1)
            data = data_aug(p, 1)
            skeletons_all_train.append(data)
            labels14_all_train.append(label_14 - 1)
            labels28_all_train.append(label_28 - 1)
            data = data_aug(p, 2)
            skeletons_all_train.append(data)
            labels14_all_train.append(label_14 - 1)
            labels28_all_train.append(label_28 - 1)
            data = data_aug(p, 3)
            skeletons_all_train.append(data)
            labels14_all_train.append(label_14 - 1)
            labels28_all_train.append(label_28 - 1)

    print(len(skeletons_all_train))

    pickle.dump(skeletons_all_train, open(os.path.join(root, 'aug4_train_skeleton.pkl'), 'wb'))
    pickle.dump([names_all_train, labels14_all_train],
                open(os.path.join(root, 'aug4_train_label_14.pkl'), 'wb'))
    pickle.dump([names_all_train, labels28_all_train],
                open(os.path.join(root, 'aug4_train_label_28.pkl'), 'wb'))

    # for line in tqdm(val_split):
    #     line = line.rstrip()
    #     g_id, f_id, sub_id, e_id, label_14, label_28, size_seq = map(int, line.split(" "))
    #     src_path = os.path.join(root, "gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt"
    #                             .format(g_id, f_id, sub_id, e_id))
    #     skeletons, num_frame = read_skeleton(src_path)
    #     skeletons = normalize_skeletons(skeletons, origin=0, base_bone=[0, 10])
    #
    #     skeletons_all_val.append(skeletons)
    #     labels14_all_val.append(label_14-1)
    #     labels28_all_val.append(label_28-1)
    #     names_all_val.append("{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id))

    # pickle.dump(skeletons_all_val, open(os.path.join(root, 'val_skeleton.pkl'), 'wb'))
    # pickle.dump([names_all_val, labels14_all_val],
    #             open(os.path.join(root, 'val_label_14.pkl'), 'wb'))
    # pickle.dump([names_all_val, labels28_all_val],
    #             open(os.path.join(root, 'val_label_28.pkl'), 'wb'))

if __name__ == '__main__':
    gendata()
    # root = 'C:/ML/dataset/HandGestureDataset_SHREC2017/val_skeleton.pkl'
    #
    # with open(root, 'rb') as f:
    #     data = pickle.load(f)
    #     print(len(data))
    # for i in range(0, len(data)):
    #     j = data[i]
    #     tmp = np.asarray(j)
    #     out = uniform_sample_np(tmp, 150)
    #     print(out.shape)

