import cv2
import pandas as pd
import os
import numpy as np
import zipfile
from urllib.request import urlretrieve

dirnames = []
for dirname in os.getcwd().split('/'):
    dirnames.append(dirname)
    if dirname == 'nia-project':
        break
DATA_DIR = "/".join(dirnames+['data'])
ICDAR_URL ='https://s3.ap-northeast-2.amazonaws.com/pai-datasets/nia-dataset/icdar2015.zip'

class OCRDataset:
    def __init__(self, dataset='icdar2015', data_type='train'):
        self.df = load_dataset(dataset, data_type)
        self.image_dir = os.path.join(DATA_DIR,
                                      "{}/{}/images/".format(dataset, data_type))
        self.filenames = self.df.filename.unique()
        self.num_data = len(self.filenames)
        self._cached_images = {}

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            filename = self.filenames[index]
            image = self._get_image(filename)
            label = self._get_label(filename)
            return image, label
        else:
            filenames = self.filenames[index]
            images = []
            labels = []
            for filename in filenames:
                images.append(self._get_image(filename))
                labels.append(self._get_label(filename))
            return images, labels

    def _get_image(self, filename):
        if not filename in self._cached_images:
            image_path = os.path.join(self.image_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._cached_images[filename] = image
        return self._cached_images[filename].copy()

    def _get_label(self, filename):
        pts = self.df.loc[self.df.filename == filename,
              "x1":"y4"].values.reshape(-1, 4, 2)
        return pts

    def show_image(self, filename):
        if isinstance(filename, int):
            filename = self.filenames[filename]

        image = self._get_image(filename)
        rects = self._get_label(filename)
        return cv2.polylines(image, rects, True, (255, 0, 0), 3)

    def shuffle(self):
        np.random.shuffle(self.filenames)


def load_dataset(dataset, data_type):
    data_path = os.path.join(DATA_DIR,
                             '{}/{}/data.csv'.format(dataset, data_type))
    if not os.path.exists(data_path):
        download_dataset(dataset)
    df = pd.read_csv(data_path, sep='\t')
    return df


def download_dataset(dataset):
    path = 'icdar2015.zip'
    urlretrieve(ICDAR_URL, path)
    with zipfile.ZipFile(path,'r') as f:
        f.extractall(os.path.join(DATA_DIR, dataset))