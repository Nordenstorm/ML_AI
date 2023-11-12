import numpy as np

def gaussian_MAP(X, y):
	D=len(X[0])
	mu=np.array(np.zeros([2,D]))
	sigma2=np.array(np.zeros([2,D]))

	Z_0=0
	Z_1=0
	for n,x in enumerate(X):

		if y[n]==0:

			mu[0,:]=mu[0,:]+x
			Z_0=Z_0+1



		else:
			mu[1,:]=mu[1,:]+x
			Z_1=Z_1+1

	mu[0,:]=mu[0,:]/Z_0
	mu[1,:]=mu[1,:]/Z_1

	for n,x in enumerate(X):

		if y[n]==0:

			sigma2[0,:]=sigma2[0,:]+((mu[0,:]-x)**2)

		else:

			sigma2[1,:]=sigma2[1,:]+((mu[1,:]-x)**2)

	sigma2[0,:]=sigma2[0,:]/(Z_0)
	sigma2[1,:]=sigma2[1,:]/(Z_0)
	sigma2=sigma2


	return mu, sigma2

def gaussian_MLE(y):

	return np.sum(y)/len(y)

def gaussian_classify(mu, sigma2, p, X):

	Y_hat=[]
	for n,x in enumerate(X):


		estimate_y_log_prob_1=np.log(p)	
		estimate_y_log_prob_0=np.log(1-p)

		deltas1=((mu[1,:]-x)**2)/(2*sigma2[1,:])
		deltas0=((mu[0,:]-x)**2)/(2*sigma2[0,:])
		exp_deltas1=np.exp(-deltas1)
		exp_deltas0=np.exp(-deltas0)
		logs_deltas1=np.log(exp_deltas1)
		logs_deltas0=np.log(exp_deltas0)

		estimate_y_log_prob_1=estimate_y_log_prob_1+np.sum(logs_deltas1)
		estimate_y_log_prob_0=estimate_y_log_prob_0+np.sum(logs_deltas0)

		

		if estimate_y_log_prob_1<estimate_y_log_prob_0:

			Y_hat.append(0)
		else:
			Y_hat.append(1)

	return Y_hat


X=np.array([[53,63,31],[1,-10,-40],[40,30,10],[-10,-15,-20]])
y=np.array([1,0,1,0])
mu,sigma2=gaussian_MAP(X,y)
print(sigma2)

p=gaussian_MLE(y)
X_test=np.array(([[58,63,31],[1,-15,-40],[40,70,10],[-10,-15,-20]]))
print((gaussian_classify(mu, sigma2, p, X_test)))
