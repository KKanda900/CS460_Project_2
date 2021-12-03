import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import sys

def visualize_problem(robot, landmarks):
    fig,ax = plt.subplots()
        
    p1 = Polygon(robot, facecolor='k')
    ax.add_patch(p1)
    
    for landmark in landmarks:
        ax.plot(landmark[0], landmark[1], 'g*')
        
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])

    plt.show()
    
robot = [[10, 10], [20, 20], [15.0, 5]]

N = 0
landmarks = []
line_counter = 0

with open(sys.argv[1]) as f:
    for line in f:
        if line_counter == 0:
            N = int(line)
        else:
            line_arr = line.split(' ')
            x = line_arr[0].replace('\n', '')
            y = line_arr[1].replace('\n', '')
            landmarks.append((float(x), float(y)))
        line_counter += 1
        
visualize_problem(robot, landmarks)