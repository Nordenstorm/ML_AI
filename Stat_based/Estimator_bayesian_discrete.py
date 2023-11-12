import numpy as np

def bayes_MAP(X,y):

	N=len(X)
	D=len(X[0])

	Theta=np.array(np.zeros([2,D]))

	#I want a code that estimates the prob P(x / Y=c) where c=(0,1)

	for n_vec,x_vec in enumerate(X):

		for d,x in enumerate(x_vec):


			if x==1:

				if y[n_vec]==0:

					Theta[0][d]=Theta[0][d]+1

				if y[n_vec]==1:
			
					Theta[1][d]=Theta[1][d]+1

	for i in range(0,len(Theta[0])):

		Z=Theta[0][i]+Theta[1][i]
		Theta[0][i],Theta[1][i]=Theta[0][i]/Z,Theta[1][i]/Z


	
	return Theta

def bayes_MLE(y):

	return np.sum(y)/len(y)


def bayes_classify(theta,p,X):

	#Theta=[2,N], y=p, X_TEST=[X_1,X_2,X_3,....]  
	# X_TEST,i=[0,0,0,1,1,1,1,1,0,0,0,0], we are to predict the y assosiated with X_test, we call this estimate \hat(y)
	Y_hat=[]
	for n_test,x_test in enumerate(X):
		estimate_y_log_prob_1=np.log(p)
		estimate_y_log_prob_0=np.log(1-p)

		for n_i,x_i in enumerate(x_test):

			if x_i==1:

				estimate_y_log_prob_1=estimate_y_log_prob_1+np.log(theta[1][n_i])

			else:

				estimate_y_log_prob_0=estimate_y_log_prob_0+np.log(theta[0][n_i])



		if estimate_y_log_prob_1>estimate_y_log_prob_0:

			Y_hat.append(0)
		else:
			Y_hat.append(1)
	return Y_hat




"""
X=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,0],[1,1,1,0],[1,1,1,0],[1,1,1,0],[1,1,1,0],[1,1,1,0]])
print("X")
print(X)
X=np.array(X)
#y=[1,1,0,0,1,1,0,1]
y=[0,1,1,1,1,0,1,0]
p=bayes_MLE(y)
Theta=bayes_MAP(X,y)
print(Theta)

X_test=[[1,1,1,0],[0,0,0,0]]



if 1==1:
	Y_hat=bayes_classify(Theta,p,X_test)

print(Y_hat)
"""

X=np.array([(10/21)*0.2, (2/21)*0.5, (9/21)*0.3])
X=X/(np.sum(X))
print(X)

