import numpy as np
import matplotlib.pyplot as plt
from config import LABELS_INFO

BISENET_COLORS = False
if BISENET_COLORS:
    np.random.seed(123)
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    palette = np.flip(palette, axis=1)
else:
    palette = np.loadtxt("dev/seg_rgbs.txt", dtype=np.uint8)
    breakpoint()

ids = [x["trainId"] for x in LABELS_INFO]
names = [x["name"] for x in LABELS_INFO]

# names = [
#    "trunks",
#    "stumps",
#    "trails",
#    "fuel",
#    "tree canopy",
#    "sky",
#    "soil",
# ]
print(palette)
print(names)

fig, axs = plt.subplots(2, 4)
for i in range(7):
    idx = ids[i]
    name = names[idx]
    color = palette[idx]
    color = np.expand_dims(np.expand_dims(color, axis=0), axis=1)
    axs[i // 4, i % 4].imshow(color)
    axs[i // 4, i % 4].set_title(name)

[(ax.set_xticks([]), ax.set_yticks([])) for ax in axs.flatten()]

plt.suptitle("Colors of classes in visualization")
plt.savefig("vis/class_colors.png")
plt.show()
