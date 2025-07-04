from torch.utils.data import Dataset, DataLoader
import torch

# Dummy dataset per test
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, num_frames=4, in_channels=2, T=287, F=597, num_classes=2):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.T = T
        self.F = F
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random input: (S, 2, T, F)
        x = torch.randn(self.num_frames, self.in_channels, self.T, self.F)
        # Random label (0 or 1)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return x, label


# Dummy dataloader
def get_dummy_dataloader(batch_size=16, shuffle=True):
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)