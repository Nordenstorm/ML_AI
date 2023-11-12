import torch
import matplotlib.pyplot as plt


def load_data():
    N = 20
    std = .7
    torch.manual_seed(5)
    x = torch.cat(
        (
            std * torch.randn(N, 2) + torch.Tensor([[2, -2]]),
            std * torch.randn(N, 2) + torch.Tensor([[-2, 2]]),
        ),
        0,
    )
    init_c = torch.Tensor([[-2, 2], [-2, 2]]) + std * torch.randn(2, 2)
    return x, init_c


def vis_cluster(c, x1, x2):
    '''
    Visualize the data and clusters.

    Argument:
        c: cluster centers [2, 2]
        x1: data points belonging to cluster 1 [#cluster_points, 2]
        x2: data points belonging to cluster 2 [#cluster_points, 2]
    '''
    # c[2, 2]
    # x1, x2: [#cluster_points, 2] where x1 and x2 belongs in different clusters
    plt.plot(x1[:, 0].numpy(), x1[:, 1].numpy(), "ro")
    plt.plot(x2[:, 0].numpy(), x2[:, 1].numpy(), "bo")
    l = plt.plot(c[:, 0].numpy(), c[:, 1].numpy(), "kx")
    plt.setp(l, markersize=10)
    plt.show()



def cat_me_k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [N, 2].
        init_c: initial centroids, shape [2, 2]. Each row is a centroid.
    
    Return:
        c: shape [2, 2]. Each row is a centroid.
    """
    #c=torch.zeros(2,2)

    if X is None:
        X, init_c = load_data()
    c=init_c
    

    for n in range(0,n_iters):
        belong=torch.zeros(len(X),2)

        for i in range(0,len(X)):

            norm_c0=torch.linalg.norm(X[i]-c[0])
            norm_c1=torch.linalg.norm(X[i]-c[1])
            if norm_c0<norm_c1:
                belong[i,0]=1
            else:
                belong[i,1]=1

        c=torch.zeros(2,2)
        sumc1=0
        sumc2=0

        for i in range(0,len(X)):

            sumc1=sumc1+belong[i,0]
            sumc2=sumc2+belong[i,1]

            c[0]=c[0]+belong[i,0]*X[i]
            c[1]=c[1]+belong[i,1]*X[i]
            
        c[0]=c[0]/sumc1
        c[1]=c[1]/sumc2
    X_blue=torch.tensor([[0,0]])
    X_red=torch.tensor([[0,0]])

    belong=torch.zeros(len(X),2)

    for i in range(0,len(X)):

        norm_c0=torch.linalg.norm(X[i]-c[0])
        norm_c1=torch.linalg.norm(X[i]-c[1])
        if norm_c0<norm_c1:
            belong[i,0]=1
        else:
            belong[i,1]=1
    for i in range(0,len(X)):

        little_x=torch.unsqueeze(X[i],axis=0)

        if belong[i,0]==1:
            X_blue=torch.cat((X_blue,little_x),dim=0)
        else:
            X_red=torch.cat((X_red,little_x),dim=0)

    X_red=X_red[1:]
    X_blue=X_blue[1:]

    return c,X_blue,X_red

x,inital_c=load_data()



c,X_blue,X_red=cat_me_k_means(x,inital_c,n_iters=4)

print(X_blue)

print(X_red)

vis_cluster(c,X_blue,X_red)