
import os
import numpy as np
import random
import cv2

class BasicDataLoader(object):
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.image_folder=image_folder
        self.image_list_file=image_list_file
        self.transform=transform
        self.shuffle=shuffle
        self.data_list=self.read_list()

    def read_list(self):
        data_list=[]
        with open(self.image_list_file) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.image_folder, line.split()[1])
                data_list.append((data_path, label_path))
        random.shuffle(data_list)
        return data_list

    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape
        assert h==h_gt, "Error"
        assert w==w_gt, "Error"
        if self.transform:
            data, label = self.transform(data, label)
        label = label[:,:,np.newaxis]  
        return data, label

    def __len__(self):
        return len(self.data_list)

    def __call__(self):
        for data_path, label_path in self.data_list:
            data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            label = cv2. imread(label_path, cv2.IMREAD_GRAYSCALE)
            #print(data.shape, label.shape)
            data, label = self.preprocess(data, label)
            yield data, label
