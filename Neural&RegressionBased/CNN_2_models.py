import Utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class DigitsConvNet(nn.Module):
    def __init__(self):

        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        '''

        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!


        self.input_conv = nn.Sequential()
        self.input_conv.add_module("Conv0", nn.Conv2d(in_channels=1, out_channels=7, kernel_size=3))
        self.input_conv.add_module("BN0", nn.BatchNorm2d(num_features=7))
        self.input_conv.add_module("Relu0", nn.ReLU(inplace=False))

        self.hidden1_conv = nn.Sequential()
        self.hidden1_conv.add_module("Conv1", nn.Conv2d(in_channels=7, out_channels=3, kernel_size=3,padding=1))
        self.hidden1_conv.add_module("BN1", nn.BatchNorm2d(num_features=3))
        self.hidden1_conv.add_module("Relu1", nn.ReLU(inplace=False))

        self.hidden2_maxpool = nn.Sequential()
        self.hidden2_maxpool.add_module("maxpool1", nn.MaxPool2d(kernel_size=2))

        self.hidden3_conv = nn.Sequential()
        self.hidden3_conv.add_module("Conv2", nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2))
        self.hidden3_conv.add_module("BN2", nn.BatchNorm2d(num_features=3))
        self.hidden3_conv.add_module("Relu2", nn.ReLU(inplace=False))

        self.output_linear = nn.Sequential()
        self.output_linear.add_module("linear1", nn.Linear(3*2*2,10))

    def forward(self, xb):


        '''
        A forward pass of your neural network.

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        #Through all the data transform i will transform xb, eventhough it changes meaning to conserve meomory
 
        xb=torch.unsqueeze(xb,1)

        xb = self.input_conv(xb)
        xb = self.hidden1_conv(xb)
        xb = self.hidden2_maxpool(xb)
        xb = self.hidden3_conv(xb)
        xb=xb.view(-1,3*2*2)
        y_hat = self.output_linear(xb)

        return y_hat

class DigitsConvNetv2(nn.Module):
    def __init__(self):


        super(DigitsConvNetv2, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        self.input_conv = nn.Sequential()
        self.input_conv.add_module("Conv0", nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3))
        self.input_conv.add_module("BN0", nn.BatchNorm2d(num_features=9))
        self.input_conv.add_module("Relu0", nn.ReLU(inplace=False))

        self.hidden1_conv = nn.Sequential()
        self.hidden1_conv.add_module("Conv1", nn.Conv2d(in_channels=9, out_channels=5, kernel_size=3,padding=1))
        self.hidden1_conv.add_module("BN1", nn.BatchNorm2d(num_features=5))
        self.hidden1_conv.add_module("Relu1", nn.ReLU(inplace=False))

        self.hidden2_maxpool = nn.Sequential()
        self.hidden2_maxpool.add_module("maxpool1", nn.MaxPool2d(kernel_size=2))

        self.hidden3_conv = nn.Sequential()
        self.hidden3_conv.add_module("Conv2", nn.Conv2d(in_channels=5, out_channels=3, kernel_size=2))
        self.hidden3_conv.add_module("BN2", nn.BatchNorm2d(num_features=3))
        self.hidden3_conv.add_module("Relu2", nn.ReLU(inplace=False))

        self.hidden4_linear = nn.Sequential()
        self.hidden4_linear.add_module("linear3", nn.Linear(3*2*2,5))
        self.hidden4_linear.add_module("linear3", nn.ReLU(inplace=False))

        self.output_linear = nn.Sequential()
        self.output_linear.add_module("linear4", nn.Linear(5,10))

    def forward(self, xb):

        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb=torch.unsqueeze(xb,1) # This is just a step of data processing

        xb = self.input_conv(xb)
        xb = self.hidden1_conv(xb)
        xb = self.hidden2_maxpool(xb)
        xb = self.hidden3_conv(xb)

        xb=xb.view(-1,3*2*2)


        xb = self.hidden4_linear(xb)
        y_hat = self.output_linear(xb)
        return y_hat

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):

    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = [Utils_to_CNN.epoch_loss(net,loss_func,train_dl)]
    test_losses = [Utils_to_CNN.epoch_loss(net,loss_func,test_dl)]


    for epoch in range(0,n_epochs):
        
        for iteration, (X_sample, Y_sample) in enumerate(train_dl):

            loss=loss_func(net.forward(X_sample),Y_sample) # Calculate loss

            # Gradient step

            optimizer.zero_grad() # Set gradient to zero
            loss.backward()       # Find new gradient 
            optimizer.step()      # Take one step with the gradient 

        train_losses.append(Utils_to_CNN.epoch_loss(net,loss_func,train_dl))
        test_losses.append(Utils_to_CNN.epoch_loss(net,loss_func,test_dl))
    return train_losses, test_losses

train,test=Utils.torch_digits()

#train[0]=torch.unsqueeze(train[0],1)
my_net=DigitsConvNetv2()
loss_func=nn.CrossEntropyLoss()
n_epochs=50
optimizer=optim.SGD(my_net.parameters(),lr=0.001)

a,b=fit_and_evaluate(my_net, optimizer, loss_func, train, test, n_epochs, batch_size=1)

print(b)


train,test=Utils.torch_digits()

#train[0]=torch.unsqueeze(train[0],1)
my_net=DigitsConvNet()
loss_func=nn.CrossEntropyLoss()
n_epochs=50
optimizer=optim.SGD(my_net.parameters(),lr=0.001)

a,b=fit_and_evaluate(my_net, optimizer, loss_func, train, test, n_epochs, batch_size=1)

print(b)

