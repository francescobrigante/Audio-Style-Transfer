import numpy as np

data = np.load(r"C:\Users\Lucia\Desktop\Uni\DL\Dataloader\stats_stft_cqt.npz")

print("Keys available in the file:", data.files)

print("Shape of stft_mean:", data["stft_mean"].shape)
print("stft_mean (preview):\n", data["stft_mean"][0][:10])

print("Shape of cqt_std:", data["cqt_std"].shape)
print("cqt_std (preview):\n", data["cqt_std"][1][:10])
