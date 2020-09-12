from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_data(dataset):
    
    if dataset == "MNIST":
        
        dataset = datasets.MNIST('./data', train=True, download=True).data

        # Setting the values in the data to be within the range [0, 1]
        dataset = dataset.numpy() / 255
        
    else:
        
        dataset = None
    
    return dataset

def peek(dataset):
    
    plt.imshow(dataset[3], cmap='gray_r')
    
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(dataset[index], cmap='gray_r')
    
    # Calculating Mean and STD for the data

    mean = np.mean(dataset)
    std = np.std(dataset)
    
    return mean, std

def transform_load(dataset, mean, std, batch_size):
    
    # Train data transformations
    train_transforms = transforms.Compose([

        # Rotating images by 7 degrees
        transforms.RandomRotation((-6.0, 6.0), fill=(1,)),

        # convert the data to torch.FloatTensor with values within the range [0.0 ,1.0]
        transforms.ToTensor(),

        # normalize the data with mean and standard deviation
        transforms.Normalize((mean,), (std,))
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    

    SEED = 1

    cuda = torch.cuda.is_available()
    print('CUDA Available?', cuda)

    # For reproducibility of results
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    
    return train_loader, test_loader