import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NVgaze as Model
from dataloader_pveye import EyeDataset as EyeDataset
from config import Config
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class PointAngleLoss:
    def __init__(self):
        self.z = torch.tensor(3.1)



    def calculate_mean_error(self, outputs, targets):
        e = (outputs[:, 0] * targets[:, 0] + outputs[:, 1] * targets[:, 1] + self.z * self.z)
        g1 = torch.sqrt(torch.square(outputs[:, 0]) + torch.square(outputs[:, 1]) + torch.square(self.z))
        g2 = torch.sqrt(torch.square(targets[:, 0]) + torch.square(targets[:, 1]) + torch.square(self.z))
        cosine = e / (g1 * g2)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        e = torch.acos(cosine) * 180 / torch.pi
        error = torch.nanmean(e)
        return error

    def loss(self, outputs, targets):

        loss = self.calculate_mean_error(outputs, targets)
        return loss

class RandomHistogramEqualization(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            image_np = np.array(img)  # Convert PIL image to NumPy array

            if image_np.ndim == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)

            equalized_img_np = cv2.equalizeHist(image_np)
            img = Image.fromarray(equalized_img_np)
        else:
            img = Image.fromarray(img)
        return img

def load_pretrained_model(model, load_path):
    try:
        model.load_state_dict(torch.load(load_path), strict=False)
        print(f"Model loaded successfully from {load_path}")
    except FileNotFoundError:
        print(f"No pretrained model found at {load_path}, starting training from scratch.")
    except Exception as e:
        print(f"Error loading the model: {e}")

def train():
    device = torch.device(f'cuda:{Config.gpu_id}')



    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert tensor or ndarray to PIL image
        # transforms.Resize(256),  # Resize image
        # transforms.CenterCrop(224),  # Center crop to 224x224
        # transforms.RandomHorizontalFlip(),  # Random horizontal flip
        # transforms.RandomVerticalFlip(),  # Random vertical flip
        # transforms.RandomRotation(15),  # Random rotation of image by ±15 degrees
    #     RandomHistogramEqualization(probability=0.5),  # 50% probability of applying histogram equalization
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize image
    ])

    train_dataset = EyeDataset(root_dir=Config.train_dir, transform=transform, eye_side=Config.eye_side)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    model = Model().to(device)

    load_pretrained_model(model, Config.load_model_path)

    criterion = nn.L1Loss()
    angle_loss_evaluator = PointAngleLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)


    model.train()
    for epoch in range(Config.epochs):
        running_loss = 0.0
        running_angle_error = 0.0  # Accumulated angle error
        for i, data in enumerate(train_loader):
            inputs, labels = data['image'].to(device), data['coordinates'].to(device)  # Move data to the specified GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                angle_error = angle_loss_evaluator.loss(outputs.detach(), labels)

            running_loss += loss.item()
            running_angle_error += angle_error.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}, Angle Error: {running_angle_error / 10:.2f} degrees')
                running_loss = 0.0
                running_angle_error = 0.0


        if (epoch + 1) % Config.save_step == 0:
            save_path = Config.save_path.format(epoch=epoch + 1)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()
