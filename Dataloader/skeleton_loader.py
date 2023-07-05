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
from torch.utils.data import Dataset, DataLoader
import torch
from random import randint, shuffle

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

    # le is None to provide a unified interface with JHMDB datagenerator
    def __call__(self, C):
        X_1, X_2, X_3 = [], [], []

        Y = []
        for i in range(len(self.data)):
            p = np.transpose(np.copy(self.data[i].squeeze(-1)),(1,2,0))
             # p.shape (frame,joint_num,joint_coords_dims)
            label = self.label[i]
            # p = normalize_skeletons(p, 0)
            p = utils.zoom(p, target_l=C.frame_l,
                           joints_num=C.joint_n, joints_dim=C.joint_d)
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



class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, label, time_len, use_data_aug):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """
        self.use_data_aug = use_data_aug
        self.data = data
        self.label = label
        self.time_len = time_len
        self.compoent_num = 22


    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        #print("ind:",ind)
        skeleton = self.data[ind]

        #hand skeleton
        skeleton = np.array(skeleton)

        if self.use_data_aug:
            skeleton = self.data_aug(skeleton)

        skeleton = torch.from_numpy(skeleton).float()
        #print(skeleton.shape)
        # label
        label = self.label[ind]

        return skeleton, label

    def data_aug(self, skeleton):

        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] *= factor
            skeleton = np.array(skeleton)
            return skeleton

        def shift(skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] += offset
            skeleton = np.array(skeleton)
            return skeleton

        def noise(skeleton):
            low = -0.1
            high = -low
            #select 4 joints
            all_joint = list(range(self.compoent_num))
            shuffle(all_joint)
            selected_joint = all_joint[0:4]

            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(self.time_len):
                    skeleton[t][j_id] += noise_offset
            skeleton = np.array(skeleton)
            return skeleton

        def time_interpolate(skeleton):
            skeleton = np.array(skeleton)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []

            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1]#d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i -1] + displace)# r*disp

            while len(result) < self.time_len:
                result.append(result[-1]) #padding
            result = np.array(result)
            return result

        # og_id = np.random.randint(3)
        aug_num = 4
        ag_id = randint(0, aug_num - 1)
        if ag_id == 0:
            skeleton = scale(skeleton)
        elif ag_id == 1:
            skeleton = shift(skeleton)
        elif ag_id == 2:
            skeleton = noise(skeleton)
        elif ag_id == 3:
            skeleton = time_interpolate(skeleton)

        return skeleton



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



