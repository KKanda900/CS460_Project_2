import sys
import numpy as np
from numpy.lib.function_base import insert
import matplotlib.pyplot as plt

'''

Things to Fix Before Submission:

    1. Simplify the predict and update equations
    2. Clean up logic in the main method
    
'''

'''
Class Definition for Kalman Filter

class Kalman_Filter:
    
    Q = R = u = A = B = H = x_k = p_k = None
    
    def __init__(self, Q, R, u):
        pass

'''

def predict(A, B, Q, u_t, x_k_1, p_k_1):
    x_k = A @ x_k_1 + B @ u_t
    p_k = A @ p_k_1 @ A.T + Q
    return x_k, p_k

def update(H, R, z, x_k, p_k):
    K = p_k @ H.T @ np.linalg.inv(H @ p_k @ H.T + R)
    x_k = p_k + K @ (z - H @ x_k)
    p_k = p_k - K @ H @ p_k
    return x_k, p_k

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
    
    # separate the data
    k = [] # k values
    u1_k_1 = [] # (u1_k−1, u2_k-1)
    u2_k_1 = []
    z = [] # (zx, zy)
    
    for i in range(0, len(data)):
        k.append(i+1)
        u1_k_1.append(float(data[i][0]))
        u2_k_1.append(float(data[i][1]))
        z.append((float(data[i][2]), float(data[i][3])))
    
    Q = [[0.0001, 0.00002], [0.00002, 0.0001]]
    R = [[0.01, 0.005], [0.005, 0.02]]
    
    u = list(np.stack((u1_k_1,u2_k_1), axis=1))

    u_t = np.array(u.pop(0))
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0], [0, 1]])

    x_k = np.array([0, 0]) 
    p_k = np.array([[0.1, 0], [0, 0.1]]) 

    pred = []

    for k_iter in range(len(k)-1):
        '''
        Steps:
            1. Predict 
            2. Update
        '''
        if k_iter > 0 and len(u) != 0:
            u_t = u.pop(0)

        x_pred, p_pre = predict(A, B, Q, u_t, x_k, p_k)
        pred.append(x_pred)
        new_measurement = z.pop(0)
        x_k, p_k = update(H, R, new_measurement, x_pred, p_pre)

    print(pred)
