import sys
import numpy as np
from numpy.lib.function_base import insert

class Kalman_Filter():
    
    def __init__(self, F, H, Q, R, u):
        self.F = F
        self.x = self.F[0][1]
        #self.x = self.F[1]
        self.y = self.F[1][2]
        #self.y = self.F[1]
        self.H = H
        self.Q = Q
        self.R = R
        self.u = u
        self.B = 0
        
    def predict(self):
        self.x = np.dot(self.F, self.x) #+ np.dot(self.B, u_vec)
        self.y = np.dot(self.F, np.dot(self.y, self.F.T)) + self.R
        
        return (self.x, self.y)
        
    def update(self, z_pt):
        S = np.dot(self.H, np.dot(self.y, self.H.T)) + R
        self.K = np.dot(np.linalg.inv(S), np.dot(self.y, self.H.T))
        #self.x = self.x + np.dot(self.K, np.subtract(z_pt, np.dot(self.H, self.x)))
        
        inner_2 = np.dot(self.H, self.x)
        inner_1 = np.subtract(z_pt, inner_2)
        inner_0 = np.dot(self.K, inner_1)
        
        self.x = self.x + inner_0
        
        self.y = np.dot(np.subtract(np.eye(2), np.dot(self.K, self.H)), self.y)

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
    u = [] # (u1_k−1, u2_k-1)
    z = [] # (zx, zy)
    
    for i in range(0, len(data)):
        k.append(i+1)
        u.append((float(data[i][0]), float(data[i][1])))
        z.append((float(data[i][2]), float(data[i][3])))
        
    # Initialize the covariance
    I = np.multiply(np.eye(3), scaler)
    
    F = np.array([[1, prev_coord[0], 0], [0, 0, prev_coord[1]]]) # State observation (x_o, y_o)
    #H = z # Observation (z_x, z_y)
    
    H = np.array([1, 0]).reshape(1, 2)
    
    kf = Kalman_Filter(F, H, Q, R, u)
    
    predictions = []
    
    for z_pt in z:
        z_pt = np.array(list(z_pt))
        pred = kf.predict()
        pred = np.dot(H, pred[0])[0]
        predictions.append(pred)
        kf.update(z_pt)