import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
from generate_segmentation import produce_segmentations
from create_instance import create_instance_mask

from pathlib import Path

CONFIG = "/home/frc-ag-1/data/SafeForestData/models/mmseg/segformer_mit-b5_512x512_160k_rui_yamaha_portugal_UAV_12_21_sharp/segformer_mit-b5_512x512_160k_rui_yamaha_portugal_UAV_12_21.py"
CHECKPOINT = "/home/frc-ag-1/data/SafeForestData/models/mmseg/segformer_mit-b5_512x512_160k_rui_yamaha_portugal_UAV_12_21_sharp/latest.pth"
VIDEO_FILE = "/home/frc-ag-1/data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/bag_4_7.avi"
OUTPUT_FOLDER = Path("/home/frc-ag-1/Downloads/semantic_seg")

class_folder = Path(OUTPUT_FOLDER, "SegmentationClass")
object_folder = Path(OUTPUT_FOLDER, "SegmentationObject")

fig, axs = plt.subplots(1, 2)
seg_generator = produce_segmentations(CONFIG, CHECKPOINT, VIDEO_FILE, opacity=1)
for i, x in enumerate(seg_generator):
    if i % 5:
        continue

    filename = f"{i:06d}_rgb.png"

    class_filepath = Path(class_folder, filename)
    object_filepath = Path(object_folder, filename)

    instance_mask = create_instance_mask(x["vis"])
    # axs[0].imshow(np.flip(x["vis"], axis=2))
    # axs[1].imshow(instance_mask)
    # plt.pause(1)
    imwrite(class_filepath, np.flip(x["vis"], axis=2))
    imwrite(object_filepath, instance_mask)
