# PVEye Dataset Baseline Method Source Code

This project contains the implementation code for the baseline methods of the PVEye dataset, suitable for model training and evaluation. Below is a detailed description of each script file:

## Configuration File
- **config.py**: This file contains key configuration parameters for model training. Please adjust the path settings according to your environment before use.

## Data Preprocessing
- **create_side_h5.py**: This script is used to create H5 files, compressing images to 224x224 pixel grayscale and saving the corresponding gaze point coordinates. Using H5 file format significantly improves the efficiency of model training.

## Data Loading
- **dataloader_PVEye.py**:
  - **EyeDataset**: Used during the training phase, this class can load all data at once and shuffle randomly, supporting training without calibration samples.
  - **EyeDataset_sep**: Used during the validation phase, it adds the functionality to load data from a single H5 file. Each H5 file contains data from a single user, one eye, in a single wearing posture, facilitating gaze calibration.

## Model Definition
- **model.py**: Implements the baseline model, inspired by the baseline method used in NVgaze.

## Training and Testing
- **train.py**: Training script, default using `transforms.Normalize(mean=[0.5], std=[0.5])` for data normalization. Other data augmentation methods have been commented out but can be uncommented and used as needed.
- **test.py**: Testing script, loads data for a single person, one eye, in a single wearing posture each time, randomly selecting 9 points for calibration (batch size is set to 512 by default).

## Additional Information
- We provide a pre-trained model `NVgaze_model.pth` and sample raw data for your convenience.
- Our code was developed in a Windows environment. Before using it, please make sure to update your file path settings in `config.py`.
- Due to the large size of our dataset, our lab is exploring suitable open-source platforms and developing reasonable open-source protocols to publish the raw images of the PVEye dataset. We plan to open-source the full dataset of raw data in the near future. Meanwhile, here is a Baidu Pan link to the complete data for 92 subjects: [Link](https://pan.baidu.com/s/1kXbU10dPWIcq5Ya2v__WpA?pwd=ykbd).

Thank you for your interest and support in our project! We look forward to your contributions and feedback.
