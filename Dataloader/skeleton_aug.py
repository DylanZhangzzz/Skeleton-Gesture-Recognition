import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pickle
from random import randint, shuffle
import random
import math
from math import sin,cos


class data_aug():

    def __init__(self, data):

        self.skeleton = data
        self.compoent_num = 22 # num of skeleton

    def scale(self):
        ratio = 0.2
        low = 1 - ratio
        high = 1 + ratio
        factor = np.random.uniform(low, high)
        video_len = self.skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(self.compoent_num):
                self.skeleton[t][j_id] *= factor
        skeleton = np.array(self.skeleton)
        return skeleton

    def shift(self):
        low = -0.1
        high = -low
        offset = np.random.uniform(low, high, 3)
        video_len = self.skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(self.compoent_num):
                self.skeleton[t][j_id] += offset
        skeleton = np.array(self.skeleton)
        return skeleton

    def noise(self):
        skeleton = np.array(self.skeleton)
        video_len = skeleton.shape[0]
        low = -0.1
        high = -low
        # select 4 joints
        all_joint = list(range(self.compoent_num))
        shuffle(all_joint)
        selected_joint = all_joint[0:6]

        for j_id in selected_joint:
            noise_offset = np.random.uniform(low, high, 3)
            for t in range(video_len):
                self.skeleton[t][j_id] += noise_offset
        skeleton = np.array(self.skeleton)
        return skeleton

    def time_interpolate(self):
        skeleton = np.array(self.skeleton)
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

    def Gaus_noise(self):
        temp = self.skeleton.copy()
        T, V, C = self.skeleton.shape
        noise = np.random.normal(0, 0.005, size=(T, V, C))
        return temp + noise

    def Rotate(self):
        axis_next = random.randint(0, 2)
        angle_next = random.uniform(0, 15)
        temp = self.skeleton.copy()
        # print('input', temp.shape)
        temp = np.expand_dims(np.array(temp).transpose((2, 0, 1)), axis=-1)  # CTVM
        # temp = np.transpose(temp, [3, 1, 2, 0])  # M, T, V, C
        # print(temp.shape)
        angle = math.radians(angle_next)
        # x
        if axis_next == 0:
            R = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]])
        # y
        if axis_next == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                          [0, 1, 0],
                          [sin(angle), 0, cos(angle)]])

        # z
        if axis_next == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                          [-sin(angle), cos(angle), 0],
                          [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.squeeze()
        # print(temp.shape)
        return temp

    def RandomHorizontalFlip(self):
        temp = self.skeleton.copy()
        # print('input', temp.shape) #(64, 22, 3)
        temp = np.expand_dims(np.array(temp).transpose((2, 0, 1)), axis=-1)  # CTVM
        # print(temp.shape)  # (3, 64, 22, 1)
        C, T, V, M = temp.shape
        if random.random() < 0.5:
            time_range_order = [i for i in range(T)]
            time_range_reverse = list(reversed(time_range_order))
            # print(time_range_reverse)
            temp = temp[:, time_range_reverse, :, :].transpose([1, 2, 3, 0])
            temp = temp.squeeze()
            return temp
        else:
            return self.skeleton.copy()

    def Shear(self):
        temp = self.skeleton.copy()
        # print('input', temp.shape) #(64, 22, 3)
        temp = np.expand_dims(np.array(temp).transpose((2, 0, 1)), axis=-1)
        # print('input', temp.shape)

        s1_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        R = np.array([[1, s1_list[0], s2_list[0]],
                      [s1_list[1], 1, s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])

        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(0, 1, 3, 2).squeeze(-1)
        # print(temp.shape)
        return temp




if __name__ == '__main__':
    p = np.random.random((64, 22, 3))
    s_aug = data_aug(p)

    s_aug.Shear()

