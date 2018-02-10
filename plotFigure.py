#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:26:30 2018

@author: etan
"""
import matplotlib.pyplot as pyplot
from matplotlib import cm

def plotFigure(a,b,c,X,Y,z):
    # Linear plane eq aX + bY + cZ = 1 (trial 1)
    Z = (1 - a*X - b*Y)/c 
    # Linear plane eq trial 2 Z = ax + by + c
    #Z = a*X + b*Y + c 
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, alpha=0.2)
    surf = ax.plot_surface(X,Y,z, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #pyplot.show()
    pyplot.xlabel('X pixels')
    pyplot.ylabel('Y pixels')
    ax.set_zlabel('Z: Depth values(mm)')
    fig.savefig("dorg_26planetest")
    
    
    
    return;