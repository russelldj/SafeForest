from imageio import imread
from create_instance import create_instance_mask
import matplotlib.pyplot as plt

data = imread("/home/frc-ag-1/Downloads/270/SegmentationObject/000000_rgb.png")
labels = create_instance_mask(data)
plt.imshow(labels)
plt.show()
