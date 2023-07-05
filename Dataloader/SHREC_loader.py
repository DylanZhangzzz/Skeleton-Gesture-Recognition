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
from prepare import aug_tools

class Sdata_generator:
    def __init__(self, data_level, label_level):
        self.data_path = Path(F"C:/ML/dataset/HandGestureDataset_SHREC2017/{data_level}_skeleton.pkl")
        self.label_path = Path(F"C:/ML/dataset/HandGestureDataset_SHREC2017/{data_level}_label_{label_level}.pkl")
        self.load_shrec_data()

    def load_shrec_data(self,):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __call__(self, C, random_choose=False, center_choose=False, aug=False):
        X_1, X_2, X_3 = [], [], []
        window_size= 150
        Y = []
        for i in range(len(self.data)):
            data_numpy = np.transpose(np.copy(self.data[i].squeeze(-1)),(1,2,0))
             # p.shape (frame,joint_num,joint_coords_dims)
            label = self.label[i]
            # print('input', data_numpy.shape)
            # p = normalize_skeletons(p, 0)
            # p = utils.zoom(p, target_l=C.frame_l,
            #                joints_num=C.joint_n, joints_dim=C.joint_d)
            if random_choose:
                data_numpy = utils.random_sample_np(data_numpy, window_size)
                # data_numpy = random_choose_simple(data_numpy, self.final_size)
            else:
                data_numpy = utils.uniform_sample_np(data_numpy, window_size)
            if center_choose:
                # data_numpy = uniform_sample_np(data_numpy, self.final_size)
                data_numpy = utils.random_choose_simple(data_numpy, C.frame_l, center=True)
            else:
                data_numpy = utils.random_choose_simple(data_numpy, C.frame_l)

            p = data_numpy

            # print(p.shape)

            # s = utils.decouple_spatial(p, C.hand_edge)
            # s = utils.get_CG(p, C) # (b, l, 231)
            # t = utils.decouple_temporal(p, 1)
            # print(p.shape,p[:, 1, :].shape)

            X_1.append(p)
            X_2.append(p)
            X_3.append(p)
            Y.append(label)

        self.X_1 = np.stack(X_1)
        self.X_2 = np.stack(X_2)
        self.X_3 = np.stack(X_3)

        self.Y = np.stack(Y)
        return self.X_1, self.X_2, self.X_3, self.Y


class SConfig():
    def __init__(self, frame_l=64):
        self.frame_l = frame_l  # the length of frames
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


if __name__ == '__main__':
    data_path = Path("C:/ML/dataset/HandGestureDataset_SHREC2017/train_skeleton.pkl")
    label_path = Path("C:/ML/dataset/HandGestureDataset_SHREC2017/train_label_28.pkl")
    # print(len(Train),len(Test))
    #
    # for i in tqdm(range(len(Train))):
    #
    #     p = np.copy(Train[i]).squeeze(-1)
    #     print(p.shape)
    C = SConfig()
    data = Sdata_generator('train', 28)
    x,y,z,l = data(C, True)

    print(x.shape, l.shape)



