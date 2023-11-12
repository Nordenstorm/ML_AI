import torch as th
import hw2_utils as utils
import matplotlib.pyplot as plt




def prepend(X):
    n,d=X.shape
    Z=th.ones(n)
    X_new=th.zeros(n,d+1)
    for i in range(0,d+1):
        if i==0:
            X_new[:,i]=Z
            
        else:
            X_new[:,i]=X[:,i-1]

    return X_new

def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    n,d=X.shape

    X=prepend(X)

    zeros=[]
    for i in range(0,d+1):
        zeros.append(0.0)
    w=th.tensor([zeros])

    for i in range(0,num_iter):
  
        pred=Y-th.matmul(X,w.T)
        grad=-(2/n)*(pred)*X
        grad=th.sum(grad, axis=0)
        w=w-grad*lrate


    return w.T


def linear_normal(X, Y):

    n,d=X.shape

    X=prepend(X)
    zeros=[]
    for i in range(0,d+1):
        zeros.append(0.0)
    
    w=th.matmul(th.matmul(th.pinverse(th.matmul(X.T , X)), X.T),(Y))

    return w


def plot_linear(X,Y,w):
    n,d=X.shape

    X_space_plot=th.linspace(0,5,1000)
    Y_space_plot=w[0]+w[1]*X_space_plot
    plt.plot(X_space_plot,Y_space_plot,label="linear normal")

    plt.plot(X,Y, 'bo')




    plt.title("Linear regression")
    plt.legend()
    plt.show()
    return plt.gcf()


# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):

    n,d=X.shape

    X=prepend(X)

    zeros=[]

    for i in range(0,d+1):
        zeros.append(0.0)

    w=th.tensor([zeros])

    for i in range(0,num_iter): 
        A=-Y*(w@(X.T)).T
        grad_tensor=-(Y*X)*(1/(2+th.exp(A)))*th.exp(A)  
        sum_vector=th.sum(grad_tensor, axis=0)
        grad=sum_vector*(1/n)

        w=w-grad*lrate

    return w.T



def logistic_vs_ols(X,Y,w,w_l):
    n,d=X.shape



    X_logistic=th.linspace(-5,5,1000)
    Y_logistic=-(w[0]+w[1]*X_logistic)/w[2]
    plt.plot(X_logistic,Y_logistic, label="logistic")
    X_linear=th.linspace(-5,5,1000)
    Y_linear=-(w_l[0]+w_l[1]*X_linear)/w_l[2]
    plt.plot(X_linear,Y_linear, label="linear")
    for i in range(0,n):
        if Y[i]==1:
            plt.plot(X[i,0],X[i,1], 'bo')
        else:
            plt.plot(X[i,0],X[i,1], 'ro')


    plt.title("Logistic regression")
    plt.legend()
    plt.show()

    return plt.gcf()


X,Y=utils.load_reg_data()

w_l=linear_normal(X, Y)
print(w_l)

plot_linear(X,Y,w_l)


