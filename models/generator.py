import keras
import numpy as np
from keras.applications.vgg16 import preprocess_input
from .data_utils import (normalize_shape, generate_output)


class DataGenerator(keras.utils.Sequence):
    "Generates Text Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=3,
                 image_shape=(736, 1280, 3),
                 fm_scale=4, shuffle=True):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_shape = list(image_shape)
        self.fm_scale = fm_scale
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        batch_dataset = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]
        images = np.zeros([self.batch_size] + self.image_shape)
        score_maps = []
        geo_maps = []
        for idx, (image, polys) in enumerate(zip(*batch_dataset)):
            image = image[:, :, ::-1]
            image = preprocess_input(image.astype(np.float))
            image, polys = normalize_shape(image, polys, self.image_shape)
            score_map, geo_map = generate_output(image, polys,
                                                 self.fm_scale)
            images[idx] = image
            score_maps.append(score_map)
            geo_maps.append(geo_map)

        score_maps = np.stack(score_maps)
        score_maps = np.expand_dims(score_maps, axis=-1)
        geo_maps = np.stack(geo_maps)

        return images, (score_maps, geo_maps)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()
