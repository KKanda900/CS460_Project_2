import sys
import numpy as np
from numpy.lib.function_base import insert
import matplotlib.pyplot as plt
import time

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

class Kalman_Filter:
    
    Q = R = u = A = B = H = x_k = p_k = None
    iterations = 0
    
    def __init__(self, Q, R, u, k, z):
        
        self.Q = Q
        self.R = R
        self.u = u
        
        self.A = np.array([[1, 0], [0, 1]])
        self.B = np.array([[1, 0], [0, 1]])
        self.H = np.array([[1, 0], [0, 1]])

        self.x_k = np.array([0, 0]) 
        self.p_k = np.array([[0.1, 0], [0, 0.1]])
        
        self.pred = []
        
        self.iterations = len(k)
        
        self.z = z
        
    def predict(self, A, B, Q, u_t, x_k_1, p_k_1):
        x_k = A @ x_k_1 + B @ u_t
        p_k = A @ p_k_1 @ A.T + Q
        return x_k, p_k
        
    def update(self, H, R, z, x_k, p_k):
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
            
    # separate the data
    k = [] # k values
    u1_k_1 = [] # (u1_k−1, u2_k-1)
    u2_k_1 = []
    z_xk = []
    z_yk = []
    z = [] # (zx, zy)
    
    for i in range(0, len(data)):
        k.append(i+1)
        u1_k_1.append(float(data[i][0]))
        u2_k_1.append(float(data[i][1]))
        z_xk.append(float(data[i][2]))
        z_yk.append(float(data[i][3]))
    
    Q = [[0.0001, 0.00002], [0.00002, 0.0001]]
    R = [[0.01, 0.005], [0.005, 0.02]]
    
    u = list(np.stack((u1_k_1,u2_k_1), axis=1))
    z = list(np.stack((z_xk,z_yk), axis=1))
    
    kf = Kalman_Filter(Q, R, u, k, z)
    
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0], [0, 1]])
    x_k = np.array([0, 0]) 
    p_k = np.array([[0.1, 0], [0, 0.1]])
    
    for k_iter in range(kf.iterations-1):
        x_pred, p_pre = kf.predict(A, B, kf.Q, kf.u.pop(0), x_k, p_k)
        new_measurement = kf.z.pop(0)
        x_k, p_k = kf.update(kf.H, kf.R, new_measurement, x_pred, p_pre)
        kf.pred.append(x_k)

    predictions = kf.pred

    #print(predictions)
    
    #print(predictions[0][0])

    x = y = []

    print(predictions)

    for i in range(len(predictions)):
        x_, y_ = 0, 0
        for j in range(len(predictions[i])):
            x_ += predictions[i][j][0]
            y_ += predictions[i][j][1]
        
        x.append(x_/2)
        y.append(y_/2)

    print(x, '\n', y)

    plt.plot(x, y) 
    plt.plot(z_xk, z_yk)
    
    plt.show()