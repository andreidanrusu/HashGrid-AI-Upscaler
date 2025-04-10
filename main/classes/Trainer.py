import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.amp import autocast, GradScaler

from ImagePreProcessing import ImagePP
from MLP import NeRFMLP
from MultiResHG2D import MRHG2D

import torch.optim as optim

torch.set_float32_matmul_precision('high')
# Balanced grid layout [(12, 16.0, 2), (14, 4.0, 8), (16,1.0, 16), (18, 0.125, 8)] - Fails with high detail pictures

class Trainer2D:

    def __init__(self, path : str, batch_size = 256):
        self.scaler = GradScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid = MRHG2D(layout=[(12, 16.0, 2), (12, 8.0, 4), (12, 1.0, 16), (14, 0.25, 16), (14, 0.125, 6)])
        self.grid = self.grid.to(self.device)
        self.mlp = NeRFMLP(input_dim=2 + self.grid.get_dimensions(), hidden_dim=64)
        self.mlp = self.mlp.to(self.device)
        self.mlp = torch.jit.script(self.mlp)
        learning_rate = 1e-3 * (batch_size / 256)
        self.optimizer = optim.Adam(list(self.grid.parameters())+ list(self.mlp.parameters()), lr= learning_rate)
        image = ImagePP(path, batch_size)
        self.image_shape = image.get_image_shape()
        self.batches =  image.split_by_batch()
        self.batches = [
            (pos_tensor.to(self.device), color_tensor.to(self.device))
            for pos_tensor, color_tensor in self.batches
        ]

    def epoch_iterations(self):
        for pos_tensor, color_tensor in self.batches:
            with autocast(device_type='cuda'):
                features = self.grid(pos_tensor)
                mlp_input = torch.cat([pos_tensor, features], dim=1)
                output = self.mlp(mlp_input)
                loss = torch.nn.functional.l1_loss(output[:, :3], color_tensor)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()


    def train(self, epoch = 500):
        for epoch in range(epoch):
            print(f"Epoch {epoch + 1}")
            self.epoch_iterations()



    def reconstruct_image(self, save_path=None):
        H, W = self.image_shape

        coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)
        pos_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            features = self.grid(pos_tensor)  # MRHG2D forward()
            mlp_input = torch.cat([pos_tensor, features], dim=1)
            output = self.mlp(mlp_input)[:, :3].clamp(0.0, 1.0)  # RGB only

        img_np = output.cpu().numpy().reshape(H, W, 3)
        img = (img_np * 255).astype(np.uint8)
        image = Image.fromarray(img)
        image_bicubic = image.resize((W, H), resample=Image.Resampling.BICUBIC)

        if save_path:
            image.save(save_path)
        else:
            plt.imshow(image_bicubic)
            plt.title("Reconstructed Image")
            plt.axis("off")
            plt.show()
