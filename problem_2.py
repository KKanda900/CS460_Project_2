import random
import numpy as np
import matplotlib.pyplot as plt

def generate_point():
    return float(random.uniform(-5,5))

def sign(x):
    sign = -999999
    
    if x == 0.0:
        sign =  0
    if x < 0.0:
        sign = -1
    if x > 0.0:
        sign =  1
        
    return sign

def generate_sample_pts(x):
    pt1 = (sign(x)/2)
    pt2 = np.sqrt(1-np.exp(-2*(x**2)/np.pi))
    return 0.5+pt1*pt2

N = 100
pts = []

def generate_points(start, end, step_size):
    xdist = (start-end)/step_size
    pts = []

    for i in range(step_size+1):
        x = end+(i*xdist)
        pts.append(x)

    return pts[::-1]

x_pts = generate_points(-5, 5, 50)

print(x_pts)

pts = []

while N > 0 and len(x_pts) != 0:
    x = x_pts.pop(0)
    x_pt = generate_sample_pts(x)
    pts.append(x_pt)
    N -= 1
  
print(len(pts))  
print(pts)

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()
