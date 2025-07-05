import os
import numpy as np
import matplotlib.pyplot as plt

# Input directories
path_piano = r"..\train_set_stats\stats_stft_cqt_piano.npz"
path_violin = r"..\train_set_stats\stats_stft_cqt_violino.npz"

stats_piano = np.load(path_piano)
stats_violin = np.load(path_violin)

print("Piano:")
for k in stats_piano:
    print(f"  ➤ {k}: shape {stats_piano[k].shape}")

print("\nViolin:")
for k in stats_violin:
    print(f"  ➤ {k}: shape {stats_violin[k].shape}")

# Preview and comparison (first channel only)
channel = 0

print("\nPreview STFT Mean:")
print("Piano:", stats_piano["stft_mean"][channel, :10])
print("Violin:", stats_violin["stft_mean"][channel, :10])

print("\nPreview CQT Std:")
print("Piano:", stats_piano["cqt_std"][channel, :10])
print("Violin:", stats_violin["cqt_std"][channel, :10])

# Visulaization (only if "plot" = True)
plot = True  

if plot:
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs[0, 0].plot(stats_piano["stft_mean"][channel], label="Piano")
    axs[0, 0].plot(stats_violin["stft_mean"][channel], label="Violino")
    axs[0, 0].set_title("STFT Mean")
    axs[0, 0].legend()

    axs[0, 1].plot(stats_piano["stft_std"][channel], label="Piano")
    axs[0, 1].plot(stats_violin["stft_std"][channel], label="Violino")
    axs[0, 1].set_title("STFT Std")
    axs[0, 1].legend()

    axs[1, 0].plot(stats_piano["cqt_mean"][channel], label="Piano")
    axs[1, 0].plot(stats_violin["cqt_mean"][channel], label="Violino")
    axs[1, 0].set_title("CQT Mean")
    axs[1, 0].legend()

    axs[1, 1].plot(stats_piano["cqt_std"][channel], label="Piano")
    axs[1, 1].plot(stats_violin["cqt_std"][channel], label="Violino")
    axs[1, 1].set_title("CQT Std")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
