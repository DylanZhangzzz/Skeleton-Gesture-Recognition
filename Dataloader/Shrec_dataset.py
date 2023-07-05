#! /usr/bin/env python
#! coding:utf-8:w
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import copy
import utils
from Dataloader import skeleton_aug

sys.path.insert(0, '..')
current_file_dirpath = Path(__file__).parent.absolute()


class SConfig():
    def __init__(self):
        self.frame_l = 64  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.class_coarse_num = 14
        self.class_fine_num = 28
        self.feat_d = 231
        self.filters = 64

def normalize_skeletons(data, origin=None):
    '''

    :param skeleton: M, T, V, C(x, y, z) 1,frame,joints, c
    :param origin: int
    :param base_bone: [int, int]
    :param zaxis:  [int, int]
    :param xaxis:  [int, int]
    :return:
    '''

    skeleton = np.transpose(np.expand_dims(np.array(data), axis=-1), [3, 0, 1, 2])
    M, T, V, C = skeleton.shape

    # print('move skeleton to begin')
    if skeleton.sum() == 0:
        raise RuntimeError('null skeleton')
    if skeleton[:, 0].sum() == 0:  # pad top null frames
        index = (skeleton.sum(-1).sum(-1).sum(0) != 0)
        tmp = skeleton[:, index].copy()
        skeleton *= 0
        skeleton[:, :tmp.shape[1]] = tmp

    if origin is not None:
        # print('sub the center joint #0 (wrist)')
        main_body_center = skeleton[0, 0, origin].copy()  # c
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)  # only for none zero frames
            skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask

    return np.transpose(skeleton, [1, 2, 3, 0]).squeeze(axis=-1)


def zoom(p, target_l=64, joints_num=25, joints_dim=3):
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            # p_new[:, m, n] = medfilt(p_new[:, m, n], 3) # make no sense. p_new is empty.
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p_copy[:, m, n], target_l/l)[:target_l]
    return p_new


def load_shrec_data(
        train_path=current_file_dirpath / Path("../data/SHREC/train.pkl"),
        test_path=current_file_dirpath / Path("../data/SHREC/test.pkl"),
):
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))
    print("Loading SHREC Dataset")
    return Train, Test



class SConfig():
    def __init__(self):
        self.frame_l = 64  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.class_coarse_num = 14
        self.class_fine_num = 28
        self.feat_d = 231
        self.filters = 64
        self.hand_edge = ((0, 1),
        (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
        (1, 18), (18, 19), (19, 20), (20, 21))


class Sdata_generator:
    def __init__(self, label_level='coarse_label'):
        self.label_level = label_level

    # le is None to provide a unified interface with JHMDB datagenerator
    def __call__(self, T, C, aug=False):
        X_1, X_2, X_3 = [], [], []

        Y = []
        for i in tqdm(range(len(T['pose']))):
            p = np.copy(T['pose'][i].reshape([-1, 22, 3]))
            # p.shape (frame,joint_num,joint_coords_dims)
            p = normalize_skeletons(p, 0)

            p = zoom(p, target_l=C.frame_l,
                     joints_num=C.joint_n, joints_dim=C.joint_d)
            # s = utils.decouple_spatial(p, C.hand_edge)
            # s = utils.get_CG(p, C) # (b, l, 231)
            # t = utils.decouple_temporal(p, 1)


            label = (T[self.label_level])[i] - 1
            # print(p.shape,p[:, 1, :].shape)

            X_1.append(p)
            X_2.append(p)
            X_3.append(p)
            Y.append(label)

            if aug is True:
                for ag_id in range(0, 4):
                    s_aug = skeleton_aug.data_aug(p)
                    if ag_id == 0:
                        skeleton = s_aug.scale()
                    elif ag_id == 1:
                        skeleton = s_aug.shift()
                    elif ag_id == 2:
                        skeleton = s_aug.noise()
                    elif ag_id == 3:
                        skeleton = s_aug.time_interpolate()
                    # print(skeleton.shape)

                    X_1.append(skeleton)
                    X_2.append(p)
                    X_3.append(p)
                    Y.append(label)


        self.X_1 = np.stack(X_1)
        self.X_2 = np.stack(X_2)
        self.X_3 = np.stack(X_3)

        self.Y = np.stack(Y)
        return self.X_1, self.X_2, self.X_3, self.Y


if __name__ == '__main__':
    Train, Test = load_shrec_data()
    C = SConfig()
    X_1, X_2,  X_3,Y = Sdata_generator('coarse_label')(Train, C, True)
    # # X_0, X_1, Y = Sdata_generator('fine_label')(Train, C, 'fine_label')
    # print(Y.shape)
    print("X_1.shape", X_1.shape)
    print("X_2.shape", X_2.shape)
    print("X_3.shape", X_3.shape)
