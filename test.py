import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NVgaze as Model
from torch.optim.lr_scheduler import MultiStepLR
from dataloader_pveye import EyeDataset as EyeDataset
from config import Config
import os
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import random

def select_samples_with_unique_labels(labels, num_samples):
    labels_rounded = torch.round(labels * 1e5) / 1e5 
    unique_labels, inverse_indices = torch.unique(labels_rounded, dim=0, return_inverse=True)
    unique_label_indices = {}
    for idx, label_idx in enumerate(inverse_indices):
        label_idx = label_idx.item()
        if label_idx not in unique_label_indices:
            unique_label_indices[label_idx] = idx
    selected_indices = list(unique_label_indices.values())

    return selected_indices

class PointAngleLoss:
    def __init__(self):
        self.z = torch.tensor(3.1, device='cuda:0')


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
        # Check if the loss is nan and return a predefined value if it is.
        if torch.isnan(loss):
            return torch.tensor(1.0, device='cuda:0')
        else:
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



def test(model, dataset, criterion, angle_loss_evaluator):
    model.eval()
    total_files_l1_loss = 0.0
    total_files_angle_error = 0.0
    files_count = 0

    for file_path in dataset.files:
        data = dataset.load_data_from_file(file_path)
        dataloader = DataLoader(data, batch_size=Config.test_batch_size, num_workers=4, shuffle=True)

        total_l1_loss = 0.0
        total_angle_error = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch['image'].to('cuda:0'), batch['coordinates'].to('cuda:0')
                outputs = model(inputs)

                # Select 9 random samples to calculate W
                selected_indices = select_samples_with_unique_labels(labels, 9)
                X_sample = outputs[selected_indices]
                labels_sample = labels[selected_indices]
                X_sample = torch.cat([X_sample, torch.ones(X_sample.size(0), 1).to('cuda:0')], dim=1)

                # Calculate W using selected samples
                XTX = X_sample.t() @ X_sample
                XTY = X_sample.t() @ labels_sample
                W = torch.inverse(XTX) @ XTY

                # Use W to predict on all data
                X = torch.cat([outputs, torch.ones(outputs.size(0), 1).to('cuda:0')], dim=1)
                outputs = X @ W

                l1_loss = criterion(outputs, labels)
                angle_error = angle_loss_evaluator.loss(outputs, labels)

                total_l1_loss += l1_loss.item() * inputs.size(0)
                total_angle_error += angle_error.item() * inputs.size(0)
                num_samples += inputs.size(0)

        avg_l1_loss = total_l1_loss / num_samples
        avg_angle_error = total_angle_error / num_samples
        print(f"File: {os.path.basename(file_path)}, L1 Loss: {avg_l1_loss:.4f}, Average Angle Error: {avg_angle_error:.2f} degrees")

        total_files_l1_loss += total_l1_loss
        total_files_angle_error += total_angle_error
        files_count += 1

    overall_avg_l1_loss = total_files_l1_loss / files_count
    overall_avg_angle_error = total_files_angle_error / files_count

    print(f"Overall Average L1 Loss: {overall_avg_l1_loss:.4f}, Overall Average Angle Error: {overall_avg_angle_error:.2f} degrees")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = EyeDataset(root_dir=Config.test_dir, transform=transform, eye_side=Config.eye_side)

    model = load_pretrained_model(Model().cuda(), Config.load_model_path)
    criterion = torch.nn.L1Loss()
    angle_loss_evaluator = PointAngleLoss()

    test(model, dataset, criterion, angle_loss_evaluator)
