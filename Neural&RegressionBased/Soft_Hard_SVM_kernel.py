
import Utils_SVM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



def linear_kernel(x, y):

    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
    
    Returns:
        a torch.float32 scalar
    '''

    return x@y

def polynomial_kernel(x, y, p=1):

    '''
    Compute the polynomial kernel function with arbitrary power p

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        p: the power of the polynomial kernel
    
    Returns:
        a torch.float32 scalar
    '''

    return (1+x@y)**p

def gaussian_kernel(x, y, sigma=1):

    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        sigma: parameter sigma in rbf kernel
    
    Returns:
        a torch.float32 scalar
    '''

    return torch.exp(-(torch.linalg.norm(x-y)**2)/(2*sigma**2))

def svm_epoch_loss(alpha, x_train, y_train, kernel=linear_kernel):

    Loss=0

    for i in range(0,len(x_train)):
        #Loss=Loss+alpha[i]
        for j in range(0,len(x_train)):
            Loss=Loss+alpha[i]/len(x_train)-(1/2)*alpha[i]*alpha[j]*y_train[i]*y_train[j]*kernel(x_train[i],x_train[j])

    '''

    Arguments:
        alpha: 1d tensor with shape (N,), alpha is the trainable parameter in our svm 
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
    
    Returns:
        a torch.float32 scalar which is the loss function of current epoch
    '''

    return -Loss

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=linear_kernel, c=None):

    '''

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is the linear kernel.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    '''
    #Create decending variable
    alpha=torch.zeros(len(x_train), requires_grad=True)
    optimizer = torch.optim.SGD([alpha], lr)

    if c==None:
        c=float("Inf")


    for i in range(0,num_iters):

        #Calculate alpha_grad first

        loss_alpha =svm_epoch_loss(alpha, x_train, y_train, kernel)
        optimizer.zero_grad()   # Clear gradients
        loss_alpha.backward()    # Back-prop
        #optimizer.step()        # Let optimizer update parameters using calculated gradients

        #Implement : alpha.grad
        #Projector
        with torch.no_grad():
            alpha=alpha-lr*alpha.grad
            alpha=torch.clamp_(alpha,0,c)

        alpha.requires_grad=True


    return alpha


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=linear_kernel):

    '''

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    idx = torch.nonzero(alpha,as_tuple=True)
    alpha_ = alpha[idx]
    x_ = x_train[idx]
    y_ = y_train[idx]
    if len(alpha_) == 0:
        return torch.zeros((x_test.shape[0],))
    id = alpha_.argmin()
    b = 1/y_[id]
    for i in range(len(alpha_)):
        b -= alpha_[i]*y_[i]*kernel(x_[i],x_[id])
    y_test = torch.zeros((x_test.shape[0],))
    for j in range(len(x_test)):
        y_test[j] = b
        for i in range(len(alpha_)):
            y_test[j] += alpha_[i]*y_[i]*kernel(x_[i],x_test[j])
    return y_test


X,Y=Utils_SVM.xor_data()


alpha=svm_solver(X, Y, 0.01, 1000,kernel=gaussian_kernel, c=10)

Y_pred=svm_predictor(alpha, X, Y, X,kernel=gaussian_kernel)

print(Y_pred)
