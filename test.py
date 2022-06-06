import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_stanfordCars_data():
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128, padding=4, padding_mode='edge'),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.StanfordCars(root='./data',split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.StanfordCars(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    temp = scipy.io.loadmat('./data/stanford_cars/devkit/cars_meta.mat')
    temp = temp['class_names']
    n = temp.size
    classes = np.empty(shape=[n], dtype=object)
    for i in range(n):
      classes[i] = temp[0,i][0]
    return {'train':trainloader, 'test':testloader, 'classes':classes}

data = get_stanfordCars_data()

print(data['train'].__dict__)
print(data['test'].__dict__)
print(data['classes'])

dataiter = iter(data['train'])
images, labels = dataiter.next();
print(images.size())

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print("Labels:" + ', '.join('%9s' % data['classes'][labels[j]] for j in range(8)))

flat = torch.flatten(images, 1)
print(images.size())
print(flat.size())

