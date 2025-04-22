import time
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.amp import autocast, GradScaler


from ImagePreProcessing import ImagePP
from MLP import NeRFMLP
from MultiResHG2D import MRHG2D

import torch.optim as optim

#Sets the matrix multiplication precision. Options are: medium, high, highest. Lower precision = higher computational speed
torch.set_float32_matmul_precision('medium')
# Balanced grid layout [(14, 8.0, 4), (16, 2.0, 8),(18, 0.5, 8), (16,0.25, 6)]

class Trainer2D:

    def __init__(self, path : str, batch_size = 256, layout=None):
        if layout is None:
            layout = [(14, 8.0, 4), (16, 2.0, 8),(18, 0.5, 8)]

        self.scaler = torch.amp.GradScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu")
        self.grid = MRHG2D(layout=layout)
        self.grid = self.grid.to(self.device)
        self.mlp = NeRFMLP(input_dim=2 + self.grid.get_dimensions(), hidden_dim=64)
        self.mlp = self.mlp.to(self.device)
        self.mlp = torch.jit.script(self.mlp)
        learning_rate = max(3e-4,1e-4 * (batch_size / 256))
        self.optimizer = optim.Adam(list(self.grid.parameters())+ list(self.mlp.parameters()), lr= learning_rate)
        image = ImagePP(path, batch_size)
        self.image_shape = image.get_image_shape()
        self.batches =  image.split_by_batch()
        self.batches = [
            (pos_tensor.to(self.device), color_tensor.to(self.device))
            for pos_tensor, color_tensor in self.batches
        ]

    def compute_loss(self, pos_tensor, color_tensor):
        features = self.grid(pos_tensor)
        mlp_input = torch.cat([pos_tensor, features], dim=1)
        output = self.mlp(mlp_input)
        loss = torch.nn.functional.mse_loss(output[:, :3], color_tensor)
        return loss

    def epoch_iterations(self):
        for pos_tensor, color_tensor in self.batches:
            if self.device.type == "cuda":
                #Uses float16 where possible to accelerate training.
                with autocast(device_type="cuda", dtype=torch.float16):
                    loss = self.compute_loss(pos_tensor, color_tensor)
            else:
                loss = self.compute_loss(pos_tensor, color_tensor)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def train(self, epoch = 100):
        start = time.time()
        print("Training started...")
        for epoch in range(epoch):
            print(f"Epoch {epoch + 1}")
            self.epoch_iterations()
        end = time.time() - start
        print(f"{epoch} epochs completed in {end:.2f} seconds.")

    def reconstruct_image(self, samples=4, save_path=None):

        H, W = self.image_shape

        coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)
        pos_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            features = self.grid(pos_tensor)
            mlp_input = torch.cat([pos_tensor, features], dim=1)
            output = self.mlp(mlp_input)[:, :3].clamp(0.0, 1.0)

        img_np = output.cpu().numpy().reshape(H, W, 3)
        img = (img_np * 255).astype(np.uint8)
        image = Image.fromarray(img)

        if samples > 1:
            upscale_size = (W * samples, H * samples)
            image = image.resize(upscale_size, resample=Image.Resampling.BICUBIC)

        if save_path:
            image.save(save_path)
            print(f"Saved bicubic-upscaled image to: {save_path}")
        else:
            plt.imshow(image)
            plt.title(f"Bicubic Upscaled ×{samples}")
            plt.axis("off")
            plt.show()