from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def build_dataloaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    trainval = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    n = len(trainval)
    n_val = 10000
    n_train = n - n_val
    train_set, val_set = random_split(trainval, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n