import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data) # return number of samples

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long) # return sample and label