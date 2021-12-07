import sys
import numpy as np
from numpy.lib.function_base import insert
import matplotlib.pyplot as plt

def predict(A, B, Q, u_t, mu_t, Sigma_t):
    predicted_mu = A @ mu_t + B @ u_t
    predicted_Sigma = A @ Sigma_t @ A.T + Q
    return predicted_mu, predicted_Sigma

def update(H, R, z, predicted_mu, predicted_Sigma):
    residual_mean = z - H @ predicted_mu
    residual_covariance = H @ predicted_Sigma @ H.T + R
    kalman_gain = predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)
    updated_mu = predicted_mu + kalman_gain @ residual_mean
    updated_Sigma = predicted_Sigma - kalman_gain @ H @ predicted_Sigma
    return updated_mu, updated_Sigma

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 5):
        print ("Four arguments required: python kalman2d.py [datafile] [x1] [x2] [lambda]")
        exit()
    
    filename = sys.argv[1] # data file
    x10 = float(sys.argv[2]) # origin x
    x20 = float(sys.argv[3]) # origin y
    prev_coord = [x10, x20]
    scaler = float(sys.argv[4]) # delta value

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    data = []
    for line in range(0, len(lines)):
        data.append(lines[line].split(' '))
        
    ''' Kalman Filter Logic '''
        
    # Initialize constants
    Q = [[0.0001, 0.00002], [0.00002, 0.0001]]
    R = [[0.01, 0.005], [0.005, 0.02]]
    
    # separate the data
    k = [] # k values
    u1_k_1 = [] # (u1_k−1, u2_k-1)
    u2_k_1 = []
    z = [] # (zx, zy)
    
    for i in range(0, len(data)):
        k.append(i+1)
        #u.append((float(data[i][0]), float(data[i][1])))
        u1_k_1.append(float(data[i][0]))
        u2_k_1.append(float(data[i][1]))
        z.append((float(data[i][2]), float(data[i][3])))

    ground_truth_states = np.stack((u1_k_1,u2_k_1), axis=1)
    u = list(ground_truth_states.copy())

    # Re-initialize the problem with the given information
    mu_0 = np.array([0, 0])
    Sigma_0 = np.array([[0.1, 0],
                        [0, 0.1]])
    u_t = np.array([1, 1]) # we assume constant control input

    A = np.array([[1, 0],
                [0, 1]])
    B = np.array([[1, 0],
                [0, 1]])
    """ Q = np.array([[0.3, 0],
                [0, 0.3]]) """
    H = np.array([[1, 0],
                [0, 1]])
    """ R = np.array([[0.75, 0],
                [0, 0.6]]) """

    mu_current = mu_0.copy()
    Sigma_current = Sigma_0.copy()

    pred_x = []
    pred_y = []

    for k in range(len(k)-1):
        '''
        Steps:
            1. Predict 
            2. Update
        '''
        if len(u) != 0:
            u_t = u.pop(0)

        predicted_mu, predicted_Sigma = predict(A, B, Q, u_t, mu_current, Sigma_current)
        pred_x.append(predicted_mu)
        pred_y.append(predicted_Sigma)
        new_measurement = z.pop(0)
        mu_current, Sigma_current = update(H, R, new_measurement, predicted_mu, predicted_Sigma)

    print(pred_x)
    print(pred_y)
