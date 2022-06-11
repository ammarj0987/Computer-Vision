# Computer-Vision

I implemented and tested multiple training models on stanford's car dataset to compare the accuracy of each model. The dataset contains 8,144 training images and 8,041 testing images of a diverse group of cars. There are 196 labels in the form of Make, Model, Year of a car. 
For more information on the dataset: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Pytorch and torchvision are used to load classes from data and train on. Numpy and scipy are used to extract data from .mat file and load it into a py dictionary.

The images are of different sizes so to make use of tensors, I resized all images to 3 x 256 x 256. Because car designs and colors are very similar to each other, taking small random crops from the image was not as accurate. Images are also horizontally flipped 50% of the time in training. For parameters, I set the batch size to 32 and num_workers to 2. This is one batch of the data:

![download](https://user-images.githubusercontent.com/105107071/173168942-d41992f3-b15b-4ed6-8c95-55a939ab81c7.png)

The first model I used is called SimpleNet. It takes the flatten image size as inputs and output is set to 196. input is connected to hidden neurons by a fully connected layer and then hidden is connected to outputs in the same way. I then used a convolutional neural network which contains three convolutional layers and then uses a fully connected layer. I also tried another model called DarkNet which contains 3x3 convolutional layers with maxpooling ending with a global average pooling and fully connected layer. The last mdoel I used is resnet which I loaded from pytorch and reinitialized the final layer with 196 outputs.

For my training function, I used the following parameter values: epochs=10, lr=0.01, momentum=0.9, decay=0.0, verbose=1. For simpleNet, I set epochs to 5 because the loss remained the same after 5 epochs. After training each model, I got the folowing graph for losses:

Red - SimpleNet

Black - DarkNet

Blue - CNN

Green - Resnet

![download](https://user-images.githubusercontent.com/105107071/173172740-b8f130cd-05b0-445a-b731-6094b11f59c8.png)

After testing the models on training and testing data, I got the folowing results for accuracy:

SimpleNet:  Training = 0.008595, Testing = 0.008705

CNN:        Training = 0.939096, Testing = 0.029598

DarkNet     Training = 0.079691, Testing = 0.055341

Resnet      Training = 0.997544, Testing = 0.830742
