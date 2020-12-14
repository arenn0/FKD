import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (GrayScale), 32 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (32, 222, 222)
        self.conv1 = nn.Conv2d(1, 32, 3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output dimensions: (32, 111, 111)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # second conv layer: 111 inputs, 32 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = 109
        # the output tensor will have dimensions: (32, 109, 109)
        # after another pool layer this becomes (32, 54, 54); .5 is rounded down
        self.conv2 = nn.Conv2d(32, 32, 3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)

        # second conv layer: 54 inputs, 32 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (32, 52, 52)
        # after another pool layer this becomes (32, 26, 26);
        self.conv3 = nn.Conv2d(32, 16, 3)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2)
        
        
        # 16 outputs * the 26*26 filtered/pooled map size
        self.fc1 = nn.Linear(16*26*26, 500)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(500, 136)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # final output
        return x
