import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10


def get_dataloader(dataset = "MNIST", batch_size = 128):
    transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))  # Correpsonds to scaling between [-1, 1] --> (x - 0.5)/0.5
            ]
        )
    
    transforms_cifar10_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scaling between [-1, 1]
            ]
        )
    
    transforms_cifar10_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scaling between [-1, 1]
            ]
        )
    if dataset == "MNIST":
        trainset = MNIST("./temp/", train=True, download=True, transform= transforms)
        testset = MNIST("./temp/", train=False, download=True, transform= transforms)
    elif dataset == "CIFAR10":
        trainset = CIFAR10("./temp/", train=True, download=True, transform= transforms_cifar10_train)
        testset = CIFAR10("./temp/", train=False, download=True, transform= transforms_cifar10_test)


    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=True)
    return train_dataloader, test_dataloader


def get_dataloader_evaluation(dataset = "MNIST", batch_size = 128):
    transforms_mnist = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            torchvision.transforms.Resize((299, 299)),  # Resize for InceptionV3
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #normalize for InceptionV3
        ]
    )

    transforms_cifar10 = torchvision.transforms.Compose(
          [
              torchvision.transforms.Resize((299, 299)),  # Resize for InceptionV3
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # CIFAR normalization
          ])
    

    if dataset == "MNIST":
        trainset = MNIST("./temp/", train=True, download=True, transform= transforms_mnist)
    elif dataset == "CIFAR10":
        trainset = CIFAR10("./temp/", train=True, download=True, transform= transforms_cifar10)


    train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)
    return train_dataloader