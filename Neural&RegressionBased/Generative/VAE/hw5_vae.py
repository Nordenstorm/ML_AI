import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hw5_utils import *




# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units. 

        self.fc1 = nn.Linear(latent_dimension, hidden_units)
        self.fc2 = nn.Linear(hidden_units, data_dimension)


    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]


        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        hidden = torch.tanh(self.fc1(z))
        p = torch.sigmoid(self.fc2(hidden))

        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if True:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        batch,latent=mu.shape
        gaussian=torch.normal(mean=torch.zeros(batch,latent))
        sample=mu+torch.sqrt(sigma_square)*gaussian
        
        return sample

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension]


        x=torch.bernoulli(p)
        
        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagonal gaussian [batch_size]

        batch_size,latent_dimension=z.shape
        logprob=torch.zeros(batch_size)
        for index_i,(z_i,mu_i,sigma_i) in enumerate(zip(z,mu,sigma_square)):

            term1=-(1/2)*torch.sum((z_i-mu_i)*(1/(sigma_i)*(z_i-mu_i)))

            term2=-(1/2)*torch.sum(torch.log(2*torch.pi*sigma_i))

            logprob[index_i]=term1+term2

        return logprob

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]

        batch_size,latent_dimension=x.shape
        logprob=torch.zeros(batch_size)

        for index_i, (x_i, p_i) in enumerate(zip(x, p)):
            log_p=0
            for index_j, (x_ij, p_ij) in enumerate(zip(x_i, p_i)):

                if x_ij==1:
                    log_p=log_p+torch.log(p_ij)

                else:
                    log_p=log_p+torch.log(1-p_ij)

            logprob[index_i]=logprob[index_i]+log_p


        return logprob

    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        batch_size,laten=sampled_z.shape
        # log_p_z(z) log probability of z under prior

        batch_size,latent_dimension=sampled_z.shape
        z_mu = torch.zeros(batch_size,latent_dimension)
        z_sigma = torch.ones(batch_size,latent_dimension)

        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        log_p = self.logpdf_bernoulli(x, p)

        elbo=torch.mean(log_p+log_p_z-log_q)

        return elbo

    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(int((num_iters)/2)):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%10 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

                # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    @torch.no_grad()
    def visualize_data_space(self):

        zs=self.sample_z(torch.zeros(10,2), torch.ones(10,2))# This is the assumed prior 

        p_x_given_z=self.decoder(zs)
        plot_pic=[]
        images=[]
        
        for i in range(0,len(p_x_given_z)):
            images.append(array_to_image(p_x_given_z[i]))
            x=array_to_image(self.sample_Bernoulli(p_x_given_z[i]))
            plot_pic.append(x)
            images.append(plot_pic[i])

        to_plot=concat_images(images, 2, 10, padding = 3)

        plt.imshow(to_plot)

        plt.show()


    def visualize_latent_space(self):

        mu, sigma_square = self.encoder(self.train_images)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'lime', 'gray', 'indigo', 'teal', 'maroon', 'azure']
        
        # Convert the tensor to a NumPy array
        my_array = self.train_labels.numpy().tolist()
        mu = mu.detach_()

        legend_dict = {}  # Dictionary to store color-label associations for legend

        for i in range(0, len(mu)):
            index_of_one = my_array[i].index(1.0)
            plt.plot(mu[i, 0], mu[i, 1], colors[index_of_one], marker="o", label=f'Label {index_of_one}')
            
            # Add color-label associations to the legend dictionary
            if colors[index_of_one] not in legend_dict:
                legend_dict[colors[index_of_one]] = f'Label {index_of_one}'

        # Create legend based on the color-label associations, organized in numerical order
        legend_entries = sorted(legend_dict.items(), key=lambda x: int(x[1].split()[1]))
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for color, label in legend_entries]
        plt.legend(handles=legend_handles, title='Color Legends', loc='upper right')

        plt.show()

    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    @torch.no_grad()
    def visualize_inter_class_interpolation(self):


        
        # TODO: Sample 3 pairs of data with different classes
        muA=torch.tensor([2.579,0])
        muB=torch.tensor([0,2.87])
        muC=torch.tensor([0,-3])

        # TODO: Encode the data in each pair, and take the mean vectors


        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        data=[]
        for i in range(0,11):
            dataAB=self.interpolate_mu(muB, muA, alpha = 0.1*i)
            data.append(dataAB)
            dataBC=self.interpolate_mu(muC, muB, alpha = 0.1*i)
            data.append(dataBC)

        data = torch.stack(data)
        p_x_given_z=self.decoder(data)
        images=[]

        for i in range(0,len(p_x_given_z)):
            images.append(array_to_image(p_x_given_z[i]))

        to_plot=concat_images(images, 2, 11)

        plt.imshow(to_plot)

        plt.show()
        

        # Concatenate these plots into one figure


      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    #vae.train()

    # visualize the latent space
    #vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()
main()