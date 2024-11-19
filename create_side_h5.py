import os
import re
from PIL import Image
import h5py
import numpy as np

def read_and_parse_files(root_folder, output_base):
    # Iterate through all subfolders in root_folder
    for subfolder_name in os.listdir(root_folder):
        base_folder = os.path.join(root_folder, subfolder_name)
        if not os.path.isdir(base_folder):
            continue  # Skip if it's not a directory

        # Create output directory
        output_folder = os.path.join(output_base, subfolder_name)
        os.makedirs(output_folder, exist_ok=True)

        items = os.listdir(base_folder)
        txt_files = [item for item in items if item.endswith('.txt')]

        # Iterate through each txt file
        for txt_file in txt_files:
            file_number = txt_file[-5:-4]  # Assuming a fixed filename format
            file_number_padded = f"{int(file_number):02d}"
            folder_path = os.path.join(base_folder, file_number_padded)

            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist.")
                continue

            # Read and parse the file, read one line out of every five
            line_counter = 0
            file_path = os.path.join(base_folder, txt_file)
            print(file_path)
            with open(file_path, 'r') as file:
                for line in file:
                    line_counter += 1
                    if line_counter % 2 != 1:
                        continue

                    matches = re.split(r'0GX:|GY:|PX:', line)
                    if len(matches) > 3:
                        id_value = matches[0].strip()
                        coordinates = [matches[1].strip().split()[0], matches[2].strip().split()[0]]

                        left_eye_folder = os.path.join(folder_path, file_number + 'L')
                        right_eye_folder = os.path.join(folder_path, file_number + 'R')
                        for eye_folder, eye_side in [(left_eye_folder, 'L'), (right_eye_folder, 'R')]:
                            eye_image_path = os.path.join(eye_folder, id_value + '.jpg')
                            if os.path.exists(eye_image_path):
                                with Image.open(eye_image_path) as img:
                                    img_gray = img.convert('L')
                                    img_resized = img_gray.resize((224,224), Image.Resampling.LANCZOS)
                                    img_data = np.array(img_resized)

                                    # Create HDF5 file for each eye side
                                    h5_filename = os.path.join(output_folder, f"{subfolder_name}_{file_number_padded}_{eye_side}.h5")
                                    with h5py.File(h5_filename, 'a') as h5f:  # Use 'a' mode to add data
                                        group_key = f"{id_value}"
                                        if group_key in h5f:
                                            continue
                                        group = h5f.create_group(group_key)
                                        group.create_dataset('image', data=img_data, compression="gzip")
                                        group.attrs['coordinates'] = coordinates
                                        group.attrs['eye'] = eye_side

if __name__ == "__main__":
    root_folder = r"D:\20_Person_Dataset"
    output_base = r"D:\PVeye\session_dataset"
    read_and_parse_files(root_folder, output_base)
