from PIL import Image
import numpy as np
import torch


class ImagePP:

    def __init__(self, path, batch_size = 256):
        image = Image.open(path).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        self.image_shape = (image_np.shape[0], image_np.shape[1])
        self.batch_size = batch_size
        self.coordinates = self.get_coordinate_vector(image_np)
        self.image_flat_colors = self.flatten_image_colors(image_np)

    def get_image_colors(self):
        return self.image_flat_colors

    def get_image_coordinates(self):
        return self.coordinates

    def get_image_shape(self):
        return self.image_shape

    @staticmethod
    def flatten_image_colors(image_array : np.array):
        return image_array.reshape(-1, 3)

    @staticmethod
    def get_coordinate_vector(image_array : np.array):
        h, w, _ = image_array.shape
        return np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2)

    def split_by_batch(self):
        nr_of_values = self.coordinates.shape[0]
        nr_of_batches = int(np.ceil( nr_of_values / self.batch_size))
        batches = []
        for i in range(nr_of_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, nr_of_values)

            coordinates_tensor = torch.tensor(self.coordinates[start:end], dtype = torch.float32)
            color_tensor = torch.tensor(self.image_flat_colors[start:end], dtype = torch.float32)

            batches.append((coordinates_tensor, color_tensor))

        return batches