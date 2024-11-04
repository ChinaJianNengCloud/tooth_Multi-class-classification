import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from mypath import Path
import pandas as pd


def save_to_excel(columns, data, file_name):
    df = pd.DataFrame(data, columns=['文件名'] + columns)
    df.to_excel(file_name, index=False)


class VideoDataset(Dataset):

    # 注意第一次要预处理数据的
    def __init__(self, dataset='oral', split='train', clip_len=16, split_dataset=False):
        self.label, self.output_dir = Path.db_dir(dataset)

        self.split = split

        self.resize_height = 512
        self.resize_width = 256

        self.fnames, labels = self._load_dataset(split_dataset)

        assert len(labels) == len(self.fnames)

        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        self.label_array = np.array(labels, dtype=int)

    def _load_dataset(self, split_dataset):
        if split_dataset:
            return self.split_dataset()
        else:
            # 直接从已生成的excel表格中加载数据
            if self.split == 'train':
                excel_path = r'E:\yinda\zhujiaqian\project\excle3\train.xlsx'
            elif self.split == 'val':
                excel_path = r'E:\yinda\zhujiaqian\project\excle3\val.xlsx'
            elif self.split == 'zcx-val':
                excel_path = r'E:\yinda\zhujiaqian\project\excle3\test.xlsx'

            df = pd.read_excel(excel_path)
            fnames = df['图像文件名'].tolist()
            label_array = df.iloc[:, 1:].to_numpy()
            return fnames, label_array

    def split_dataset(self):
        dir_list = os.listdir(self.output_dir)
        whole_label = pd.read_excel(self.label)

        idx_list = [x for x in range(len(dir_list))]

        zcx_val_size = int(len(idx_list) * 0.2)
        train_size = int(len(idx_list) * 0.8 * 0.8)

        zcx_val_idx = np.random.choice(idx_list, zcx_val_size, replace=False)

        remaining_idx = np.delete(idx_list, zcx_val_idx)

        train_idx = np.random.choice(remaining_idx, train_size, replace=False)

        train_idx_in_remaining = np.array([np.where(remaining_idx == i)[0][0] for i in train_idx])

        val_idx = np.delete(remaining_idx, train_idx_in_remaining)

        columns = [col for col in whole_label.columns if col != "图像文件名"]
        fnames, labels = [], []

        if self.split == 'train':
            print("训练集划分：")
            for subject in tqdm([dir_list[x] for x in train_idx], desc="Processing Subjects"):
                fnames.append(os.path.join(self.output_dir, subject))
                one_hot_labels = [int(whole_label[col][whole_label['图像文件名'] == subject].values[0]) if len(whole_label[col][whole_label['图像文件名'] == subject].values) > 0 else 0 for col in columns]
                labels.append(one_hot_labels)
            train_subjects = [dir_list[x] for x in train_idx]
            print('开始保存训练数据')
            train_data = whole_label[whole_label['图像文件名'].isin(train_subjects)]
            train_data.to_excel(r'/root/autodl-tmp/output/train_dataset.xlsx', index=False)
            print('train保存结束')
        elif self.split == 'val':
            print("内部测试集划分：")
            for subject in tqdm([dir_list[x] for x in val_idx], desc="Processing Subjects"):
                fnames.append(os.path.join(self.output_dir, subject))
                one_hot_labels = [int(whole_label[col][whole_label['图像文件名'] == subject].values[0]) for col in columns]
                labels.append(one_hot_labels)
            print('开始保存内部测试数据')
            val_subjects = [dir_list[x] for x in val_idx]
            val_data = whole_label[whole_label['图像文件名'].isin(val_subjects)]
            val_data.to_excel(r'/root/autodl-tmp/output/val_dataset.xlsx', index=False)
            print('内部测试集结束')
        elif self.split == 'zcx-val':
            print("外部验证集划分：")
            for subject in tqdm([dir_list[x] for x in zcx_val_idx], desc="Processing Subjects"):
                fnames.append(os.path.join(self.output_dir, subject))
                one_hot_labels = [int(whole_label[col][whole_label['图像文件名'] == subject].values[0]) for col in columns]
                labels.append(one_hot_labels)
            print('开始保存zcx-val数据')
            zcx_val_subjects = [dir_list[x] for x in zcx_val_idx]
            zcx_val_data = whole_label[whole_label['图像文件名'].isin(zcx_val_subjects)]
            zcx_val_data.to_excel(r'/root/autodl-tmp/output/zcx_val_dataset.xlsx', index=False)
            print('zcx-val结束')

        return fnames, labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_path = os.path.join(self.output_dir, self.fnames[index])
        img = cv2.imread(img_path) 
        img = cv2.resize(img, (self.resize_width, self.resize_height))

        img = self.to_tensor(img)

        labels = np.array(self.label_array[index])
        return torch.from_numpy(img), torch.from_numpy(labels)

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame = frame.astype(np.float32)
            frame /= 255.0
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((2, 0, 1))
