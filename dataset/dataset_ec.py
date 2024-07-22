import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os

class ECdataset(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform

        NAME_COLUMN = 0
        LABEL_COLUMN = 1

        if self.train:
            dataset = pd.read_csv(os.path.join(self.root, 'labels/ec_train.csv'), sep=',')
        else:
            dataset = pd.read_csv(os.path.join(self.root, 'labels/ec_val.csv'), sep=',')

        if self.dataidxs is not None:
            file_names = np.array(dataset.iloc[:, NAME_COLUMN].values)[self.dataidxs]
            target = np.array(dataset.iloc[:,LABEL_COLUMN].values)[self.dataidxs]
        else:
            file_names = dataset.iloc[:, NAME_COLUMN].values
            target = dataset.iloc[:,LABEL_COLUMN].values

        self.file_paths = []
        self.targets = []

        for f,t in zip(file_names, target):
            path = os.path.join(self.root, 'face_frames/'+f+"_aligned", "face_det_000000.bmp")
            if os.path.exists(path):
                self.file_paths.append(path)
                self.targets.append(t)


    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.targets

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        target = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target