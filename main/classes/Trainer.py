import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from ImagePreProcessing import ImagePP
from MLP import NeRFMLP
from HashGrid2D import HashGrid2D

import torch.optim as optim

class Trainer2D:

    def __init__(self, path : str, batch_size = 256):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid = HashGrid2D()
        self.mlp = NeRFMLP(input_dim=2 + 4)
        self.optimizer = optim.Adam(list(self.grid.parameters())+ list(self.mlp.parameters()), lr= 1e-3)
        image = ImagePP(path, batch_size)
        self.image_shape = image.get_image_shape()
        self.batches =  image.split_by_batch()
        self.initialize_trainer()

    def initialize_trainer(self):
        for pos_tensor, color_tensor in self.batches:

            features = torch.stack([self.grid((x.item(), y.item())) for x, y in pos_tensor])  # [B, 4]

            mlp_input = torch.cat([pos_tensor, features], dim=1)

            output = self.mlp(mlp_input)

            predicted_rgb = output[:, :3]
            target_rgb = color_tensor

            loss = torch.nn.MSELoss()(predicted_rgb, target_rgb)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    def train(self, epoch = 300):
        for epoch in range(epoch):

            for pos_tensor, color_tensor in self.batches:

                features = torch.stack([self.grid((x.item(), y.item())) for x, y in pos_tensor])  # [B, 4]

                mlp_input = torch.cat([pos_tensor, features], dim=1)

                output = self.mlp(mlp_input)

                loss = torch.nn.MSELoss()(output[:,:3], color_tensor)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
            print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")



    def reconstruct_image(self, save_path = None):

        H, W = self.image_shape

        coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)
        pos_tensor = torch.tensor(coords, dtype=torch.float32)

        features = torch.stack([self.grid((x.item(), y.item())) for x, y in pos_tensor])
        mlp_input = torch.cat([pos_tensor, features], dim=1)
        with torch.no_grad():
            output = self.mlp(mlp_input)[:, :3].clamp(0.0, 1.0)  # [N, 3], RGB only

        img_np = output.cpu().numpy().reshape(H, W, 3)
        img = (img_np * 255).astype(np.uint8)
        image = Image.fromarray(img)

        if save_path:
            image.save(save_path)
        else:
            plt.imshow(img)
            plt.title("Reconstructed Image")
            plt.axis("off")
            plt.show()