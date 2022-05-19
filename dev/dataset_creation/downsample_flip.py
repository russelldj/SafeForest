from safeforest.dataset_generation.file_utils import get_files
import numpy as np
from imageio import imread, imwrite

STEM = "data_mapping_velodyne_imu_2022-04-04-10"
# STEM = "data_mapping_velodyne_imu_2022-04-04-08"
FOLDER = f"/home/frc-ag-1/data/SafeForestData/datasets/portugal_UAV_4_22/derived/{STEM}"
OUTPUT_FOLDER = f"/home/frc-ag-1/data/SafeForestData/datasets/portugal_UAV_4_22/derived/{STEM}_images"
START_INDEX = 0
files = get_files(FOLDER, "*.png")
downsampled_files = files[START_INDEX::10]
output_files = [
    str(f).replace(FOLDER, OUTPUT_FOLDER).replace("left", STEM)
    for f in downsampled_files
]
for in_f, out_f in zip(downsampled_files, output_files):
    img = imread(in_f)
    img = np.flip(img, (0, 1))
    print(out_f)
    imwrite(out_f, img)

