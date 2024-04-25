import numpy as np
from numpy import sqrt, pi, sin, cos

def Xrot(theta):
    return np.array([[1,                         0,                          0],
                     [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                     [0, np.sin(np.radians(theta)),  np.cos(np.radians(theta))]])

def Yrot(theta):
    return np.array([[np.cos(np.radians(theta)),  0, np.sin(np.radians(theta))],
                     [0,                          1,                         0],
                     [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]])

def Zrot(theta):
    return np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
                     [np.sin(np.radians(theta)),  np.cos(np.radians(theta)), 0],
                     [0,                                                  0, 1]])

O = np.array([[0, 0, 0]]).T  # base center
dcm = 0.200  # distance from base center to pan mtr
l_base = dcm * sqrt(3)
l_link = 0.400

Fnet = 50 * 9.81
a = 1 / 3.
b = 1 / 3.
c = 1 - (a + b)
f1 = np.array([[0, 0, Fnet * a]]).T
f2 = np.array([[0, 0, Fnet * b]]).T
f3 = np.array([[0, 0, Fnet * c]]).T

theta_lift = np.array([45, 45, 45])*pi/180
theta_pan = np.array([0, 0, 0])*pi/180

H_OF1  = np.vstack((np.hstack((Zrot(theta_pan[0]), np.array([[0,1/sqrt(3)*l_base,0]]).T)), np.array([[0,0,0,1]])))
H_F1F2 = np.vstack((np.hstack((Xrot(0), np.array([[0,l_link*sin(theta_lift[0]),0]]).T)), np.array([[0,0,0,1]])))

H_OL1 = np.vstack((np.hstack((Zrot(theta_pan[1] + 30), np.array([[-1/2*l_base,-1/2/sqrt(3)*l_base,0]]).T)), np.array([[0,0,0,1]])))
H_L1L2 = np.vstack((np.hstack((Yrot(0), np.array([[-l_link*sin(theta_lift[1]),0,0]]).T)), np.array([[0,0,0,1]])))

H_OR1 = np.vstack((np.hstack((Zrot(theta_pan[2] - 30), np.array([[1/2*l_base,-1/2/sqrt(3)*l_base,0]]).T)), np.array([[0,0,0,1]])))
H_R1R2 = np.vstack((np.hstack((Yrot(0), np.array([[l_link*sin(theta_lift[2]),0,0]]).T)), np.array([[0,0,0,1]])))

FK_F = np.dot(H_OF1, H_F1F2)
FK_L = np.dot(H_OL1, H_L1L2)
FK_R = np.dot(H_OR1, H_R1R2)

F1 = H_OF1[:3, 3]
L1 = H_OL1[:3, 3]
R1 = H_OR1[:3, 3]

F2 = FK_F[:3, 3]
L2 = FK_L[:3, 3]
R2 = FK_R[:3, 3]

tau_net = np.cross(F2.flatten(), f1.flatten()) + np.cross(L2.flatten(), f2.flatten()) + np.cross(R2.flatten(), f3.flatten())
xcm = tau_net[1] / (-Fnet)
ycm = tau_net[0] / Fnet
pcm = np.array([[xcm, ycm, 0]]).T  # new cm with init condition
