import torch
from torchvision import datasets, transforms

data_dir = 'Cat_Dog_data/train'
transforms = transforms.Compose([transforms.Resize(225),
                                transforms.CenterCrop(224),
                               transforms.ToTensor()])

dataset = datasets.ImageFolder(data_dir, transform=transforms)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Data Augmentation
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

data_iter = iter(train_loader)
