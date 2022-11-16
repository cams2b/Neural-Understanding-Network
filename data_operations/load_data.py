import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from config import config


class data_preprocess():
    def __init__(self, config, shuffle=False, train_img=None, train_gt=None, test_img=None, test_gt=None):
        self.config = config
        self.images = None
        self.gt = None
        self.shuffle = shuffle
        self.train_imgs = None
        self.train_gt = None
        self.val_imgs = None
        self.val_gt = None
        if config.split_data:
            self.load_data()
            self.split_data()
        else:
            self.load_split_data()
        self.encode()
        if config.mode != 'graph_visualization':
            self.oversample()
        if config.num_classes != 1:
            self.one_hot()


    def load_split_data(self):
        train_df = pd.read_excel(self.config.train_path)
        val_df = pd.read_excel(self.config.val_path)
        if self.shuffle:
            train_df = train_df.sample(frac=1)
        self.train_imgs = train_df['image'].values
        self.train_gt = train_df['groundtruth'].values
        self.val_imgs = val_df['image'].values
        self.val_gt = val_df['groundtruth'].values

    def load_data(self):
        print('[INFO] loading file paths')
        df = pd.read_excel(self.config.data)
        self.images = df['image'].values
        self.gt = df['groundtruth'].values

    def split_data(self):
        train_x, val_x, train_y, val_y = train_test_split(self.images, self.gt, test_size=self.config.validation_size,
                                                          random_state=1945)

        self.train_imgs = train_x
        self.train_gt = train_y
        self.val_imgs = val_x
        self.val_gt = val_y

    def encode(self):
        self.train_gt = self.train_gt.reshape(-1, 1)
        self.val_gt = self.val_gt.reshape(-1, 1)
        le = LabelEncoder()
        le.fit(self.train_gt)
        self.train_gt = le.transform(self.train_gt)
        self.val_gt = le.transform(self.val_gt)

    def one_hot(self):
        self.train_gt = self.train_gt.reshape(-1, 1)
        self.val_gt = self.val_gt.reshape(-1, 1)
        ohe = OneHotEncoder()
        ohe.fit(self.train_gt)
        self.train_gt = ohe.transform(self.train_gt).toarray()
        self.val_gt = ohe.transform(self.val_gt).toarray()

    def oversample(self):
        ros = RandomOverSampler(random_state=5)
        temp_train = np.array([self.train_imgs, self.train_imgs], dtype=object)
        temp_train = np.transpose(temp_train)
        x, y = ros.fit_resample(temp_train, self.train_gt)
        self.train_imgs = x[:, 0]
        self.train_gt = y

        temp_val = np.array([self.val_imgs, self.val_imgs], dtype=object)
        temp_val = np.transpose(temp_val)
        x, y = ros.fit_resample(temp_val, self.val_gt)
        self.val_imgs = x[:, 0]
        self.val_gt = y





