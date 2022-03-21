import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./dev/report.mplstyle")
real_images_finetuned = np.array([0, 30, 60, 91, 121])
real_images_trained = np.array([7, 15, 22, 30, 60, 91, 121])

RUI_ACCURACY = 0.5217762936827957
RUI_MIOU = 0.29298355970960777

FINETUNED_ACCURACY = np.array(
    [0.7032859403001792, 0.7127181339605735, 0.7026111671146953, 0.7336390779009857]
)
FINETUNED_MIOUS = np.array(
    [0.47127527106419986, 0.464863512013866, 0.4529522387161039, 0.5039886115588187]
)

ALL_FINETUNED = np.concatenate(([RUI_ACCURACY], FINETUNED_ACCURACY), axis=0)
ALL_FINETUNED_MIOUS = np.concatenate(([RUI_MIOU], FINETUNED_MIOUS), axis=0)

FROM_SCRATCH_ACCURACY = np.array(
    [
        0.7869879505921688,
        0.83648184306015,
        0.8347155252497249,
        0.8361715949820788,
        0.8554268173163082,
        0.8476031866039426,
        0.8591173485103046,
    ]
)
FROM_SCRATCH_MIOUS = np.array(
    [
        0.5860451246993091,
        0.6727075558580163,
        0.6621174936695764,
        0.6713446000257459,
        0.7037505405966754,
        0.6881875029040235,
        0.7109844812240144,
    ]
)


plt.plot(
    real_images_finetuned[1:],
    FINETUNED_ACCURACY,
    "C0",
    label="Real with synthetic pretraining",
)
plt.plot(real_images_finetuned[1:], FINETUNED_ACCURACY, "C0o")
plt.plot(real_images_trained, FROM_SCRATCH_ACCURACY, "C1", label="Only real data")
plt.plot(real_images_trained, FROM_SCRATCH_ACCURACY, "C1o")

plt.scatter(0, RUI_ACCURACY, label="Only synthetic")

plt.xlabel("Real training images")
plt.ylabel("Test accuracy")
plt.legend()
plt.savefig("/home/frc-ag-1/figures/synthetic_experiments.png")
plt.show()

plt.plot(
    real_images_finetuned[1:],
    FINETUNED_MIOUS,
    "C0",
    label="Real with synthetic pretraining",
)
plt.plot(real_images_finetuned[1:], FINETUNED_MIOUS, "C0o")

plt.plot(real_images_trained, FROM_SCRATCH_MIOUS, "C1", label="Only real data")
plt.plot(real_images_trained, FROM_SCRATCH_MIOUS, "C1o")
plt.scatter(
    0, RUI_MIOU, label="Synthetic only"
)  # horizontal alignment can be left, right or center
plt.xlabel("Number of real training images")
plt.ylabel("Test mIoU")
plt.legend()
plt.savefig("/home/frc-ag-1/figures/synthetic_experiments_mious.png")
plt.show()
