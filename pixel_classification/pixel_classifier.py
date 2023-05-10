'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
#from generate_rgb_data import read_pixels

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    """
    folder = 'pixel_classification/data/training'
    X1 = read_pixels(folder+'/red', verbose = True)
    X2 = read_pixels(folder+'/green')
    X3 = read_pixels(folder+'/blue')
    y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0], 3)
    X_train, y_train = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))

    theta = np.zeros((3,1))
    mu = np.zeros((3,3))
    sigma = np.zeros((3,3))

    for l in range(0, 3):
      for k in range(0,3):

        theta[k] = (np.count_nonzero(y_train == k+1))/3694

        mu[k,l] = np.sum((np.extract(y_train == k+1, X_train[:,l])))/(np.count_nonzero(y_train == k+1))   #(3,3)
       
        x_mu = X_train[:,l] - mu[k,l]   #broadcast
        x_mu_sq = (np.square(x_mu))
        num = np.sum(np.extract(y_train == k+1, x_mu_sq))
        den = np.count_nonzero(y_train == k+1)
        sigma[k,l] = np.sqrt(num/den)    #(3,3)
        
    print(theta, theta.shape)
    print(mu, mu.shape)
    print(sigma, sigma.shape)

    # Just a random classifier for now
    # Replace this with your own approach
    """

    theta = np.array([[0.36599892], [0.3245804], [0.30942068]])
    mu = np.array([[0.75250609, 0.34808562, 0.34891229], [0.35060917, 0.73551489, 0.32949353], [0.34735903, 0.33111351, 0.73526495]])
    sigma = np.array([[0.19250785, 0.24893512, 0.24904327], [0.23608183, 0.18650986, 0.23668942], [0.23353291, 0.23839738, 0.18905186]])
    theta_log = np.log(1/np.square(theta))    #(3,1)
    sigma_log = np.log(np.square(sigma))

    X_mu_sigma = np.zeros((X.shape[0], 3, 3))   #(83,3,3) = (n,k,l)
    #print(X_mu_sigma)
    
    for k in range(0,3):
      for l in range(0,3):
        for i in range(0,X.shape[0]):

          X_mu_sigma[i,k,l] = (np.square(X[i,l] - mu[k,l]))/np.square(sigma[k,l])

    sigma_log_sum = np.sum(sigma_log, axis = 1, keepdims=True)
    X_mu_sigma_sum = np.sum(X_mu_sigma, axis=2)   #(83,3) = (n,k)
    gradSum = np.transpose(theta_log) + np.transpose(sigma_log_sum) + X_mu_sigma_sum    #(83,3) = (n,k)
    #print(gradSum.shape)
    y = np.zeros((X.shape[0],1))
    for j in range(0, X.shape[0]):
      y[j] = np.argmin(gradSum[j,:]) + 1
      #print(y[j])

    #print(y)
  
    # YOUR CODE BEFORE THIS LINE
    ################################################################

    return y