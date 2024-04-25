from casadi import *
import numpy as np

pi = np.pi
lbase = 0.200*sqrt(3)
llink = 0.400*sin(pi/4)

xcm, ycm = 0.09656854249492386, 0
Wnet = 50*9.81

## pan motor pos
F1x = 0;          F1y = 1/sqrt(3)*lbase
L1x = -1/2*lbase; L1y = -1/2/sqrt(3)*lbase
R1x = 1/2*lbase;  R1y = -1/2/sqrt(3)*lbase

## pan angle variable
x1 = SX.sym('x1'); x2 = SX.sym('x2'); x3 = SX.sym('x3')
x = vertcat(x1,x2,x3)

## wheel center pos
F2x = -llink*sin(x[0]);                  F2y = 1/sqrt(3)*lbase + llink*cos(x[0])
L2x = -1/2*lbase - llink*sin(pi/3-x[1]); L2y = -1/2/sqrt(3)*lbase - llink*cos(pi/3-x[1])
R2x = 1/2*lbase + llink*cos(pi/6-x[2]);  R2y = -1/2/sqrt(3)*lbase - llink*sin(pi/6-x[2])

fF2x = Function('fF2x', [x], [F2x]);     fF2y = Function('fF2y', [x], [F2y])
fL2x = Function('fL2x', [x], [L2x]);     fL2y = Function('fL2y', [x], [L2y])
fR2x = Function('fR2x', [x], [R2x]);     fR2y = Function('fR2y', [x], [R2y])

## grf of wheel center pos / objective
grf1 = ((xcm-R2x)-(ycm-R2y)*(L2x-R2x)/(L2y-R2y))*Wnet / ((F2x-R2x)-(F2y-R2y)*(L2x-R2x)/(L2y-R2y))
grf2 = ((xcm-R2x)-(ycm-R2y)*(F2x-R2x)/(F2y-R2y))*Wnet / ((L2x-R2x)-(L2y-R2y)*(F2x-R2x)/(F2y-R2y))
grf3 = ((xcm-F2x)-(ycm-F2y)*(L2x-F2x)/(L2y-F2y))*Wnet / ((R2x-F2x)-(R2y-F2y)*(L2x-F2x)/(L2y-F2y))

fgrf1 = Function('fgrf1', [x], [grf1])
fgrf2 = Function('fgrf2', [x], [grf2])
fgrf3 = Function('fgrf3', [x], [grf3])

obj = (grf1-grf2)**2 + (grf2-grf3)**2 + (grf3-grf1)**2

## constraints
g1 = grf1 + grf2 + grf3 - Wnet
g2 = grf1*F2x + grf2*L2x + grf3*R2x - xcm*Wnet
g3 = grf1*F2y + grf2*L2y + grf3*R2y - ycm*Wnet
g = vertcat(g1,g2,g3)

## NLP
nlp = {'x': x, 'f': obj, 'g': g}

## Pick an NLP solver
MySolver = "ipopt"
#MySolver = "worhp"
#MySolver = "sqpmethod"

## Solver options
opts = {}
if MySolver=="sqpmethod":
  opts["qpsol"] = "qpoases"
  opts["qpsol_options"] = {"printLevel":"none"}

## Init guess / bounds
arg = {}
arg['x0'] = [0, 0, 0]
arg['lbx'] = [-np.pi/6, -np.pi/6, -np.pi/6]
arg['ubx'] = [ np.pi/6,  np.pi/6,  np.pi/6]

## Allocate a solver
solver = nlpsol("solver", MySolver, nlp, opts)

## Solve the NLP
soln = solver(**arg)


################ plot
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

pan_pt = np.array([[F1x, F1y], [L1x, L1y], [R1x, R1y]])
wheel_pt = np.array([
      [float(fF2x(soln['x'])), float(fF2y(soln['x']))],
      [float(fL2x(soln['x'])), float(fL2y(soln['x']))],
      [float(fR2x(soln['x'])), float(fR2y(soln['x']))]])

sz = 100
fig, ax = plt.subplots()

grf1, grf2, grf3 = float(fgrf1(soln['x'])), float(fgrf2(soln['x'])), float(fgrf3(soln['x']))
title = 'F1= %.2fN / F2= %.2fN / F3= %.2fN' % (grf1, grf2, grf3)
plt.title(title)

## leg
for i in range(0,3):
      plt.plot([pan_pt[i,0], wheel_pt[i,0]], [pan_pt[i,1], wheel_pt[i,1]], color='black')

## wheel
plt.plot([wheel_pt[0, 0], wheel_pt[1, 0]], [wheel_pt[0, 1], wheel_pt[1, 1]], color='lightblue')
plt.plot([wheel_pt[1, 0], wheel_pt[2, 0]], [wheel_pt[1, 1], wheel_pt[2, 1]], color='lightblue')
plt.plot([wheel_pt[2, 0], wheel_pt[0, 0]], [wheel_pt[2, 1], wheel_pt[0, 1]], color='lightblue')

plt.scatter(wheel_pt[0,0], wheel_pt[0,1], color='red', s=sz, label='leg1')
plt.scatter(wheel_pt[1,0], wheel_pt[1,1], color='green', s=sz, label='leg2')
plt.scatter(wheel_pt[2,0], wheel_pt[2,1], color='blue', s=sz, label='leg3')

## base
triangle = Polygon(pan_pt, closed=True, color='lightgray')
ax.add_patch(triangle)
plt.scatter(0, 0, marker='o', facecolor='none', edgecolor='k', s=sz, label='center')
plt.scatter(pan_pt[:,0], pan_pt[:,1], color='k', s=sz)

## cm
plt.scatter(xcm, ycm, color='magenta', s=sz, label='CM')

ax.set_aspect('equal', adjustable='box')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
k = 0.65; plt.xlim(-k, k); plt.ylim(-k, k)
plt.show()
plt.close()