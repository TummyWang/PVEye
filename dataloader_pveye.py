from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import h5py
import torch

class EyeDataset(Dataset):
    def __init__(self, root_dir, eye_side='both', transform=None):
        self.root_dir = root_dir
        self.eye_side = eye_side
        self.transform = transform
        self.files = []
        self.index_map = []

        for dp, dn, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.h5'):
                    file_path = os.path.join(dp, f)
                    self.files.append(file_path)
                    with h5py.File(file_path, 'r') as file:
                        for group_key in file.keys():
                            if self.eye_side in ['L', 'R']:
                                if file[group_key].attrs['eye'] == self.eye_side:
                                    self.index_map.append((file_path, group_key))
                            elif self.eye_side == 'both':
                                self.index_map.append((file_path, group_key))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, group_key = self.index_map[idx]
        with h5py.File(file_path, 'r') as file:
            image_data = file[group_key]['image'][()]
            coordinates = file[group_key].attrs['coordinates']

        image = image_data
        coordinates = torch.tensor(coordinates.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'coordinates': coordinates}

class EyeDataset_sep(Dataset):
    def __init__(self, root_dir, eye_side='both', transform=None):
        self.root_dir = root_dir
        self.eye_side = eye_side
        self.transform = transform
        self.files = []

        for dp, dn, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.h5'):
                    if self.eye_side == 'L' and '_L.h5' in f:
                        file_path = os.path.join(dp, f)
                        self.files.append(file_path)
                    elif self.eye_side == 'R' and '_R.h5' in f:
                        file_path = os.path.join(dp, f)
                        self.files.append(file_path)
                    elif self.eye_side == 'both':
                        file_path = os.path.join(dp, f)
                        self.files.append(file_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        return self.load_data_from_file(file_path)

    def load_data_from_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            dataset = []
            for group_key in file.keys():
                image_data = file[group_key]['image'][()]
                coordinates = file[group_key].attrs['coordinates']
                image = image_data
                coordinates = torch.tensor(coordinates.astype('float32'))

                if self.transform:
                    image = self.transform(image)

                dataset.append({'image': image, 'coordinates': coordinates})
        return dataset

if __name__ == "__main__":
    # Transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Dataset and DataLoader instantiation
    dataset = EyeDataset(root_dir='D:\PVeye\train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Iterate through DataLoader
    for i, batch in enumerate(dataloader):
        print(f'Batch {i}, Image shape: {batch["image"].shape}, Coordinates: {batch["coordinates"]}')
        if i == 3:  # Limiting the output for demonstration purposes
            break
