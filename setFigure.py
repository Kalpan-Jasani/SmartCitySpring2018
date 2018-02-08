#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:07:02 2018

@author: etan
"""
import matplotlib.pyplot as pyplot
from matplotlib import cm
import numpy


def setFigure ():
    
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    # generate x, y and z coordinates
    #X = numpy.arange(0, 620) #512
    #Y = numpy.arange(0, 460) #424
    #z = numpy.loadtxt("24in_RealCrack1.txt", delimiter=" ")
    #z = numpy.loadtxt("frame4190.txt", delimiter=" ")

    z = numpy.loadtxt("d_org130.txt", delimiter="\t")

    # to remove the edges
    #z = z[7:-7, 7:-7]
    #z = z[:,420:-115]

    # No hardcoding
    x,y = z.shape
    X = numpy.arange(0, y)
    Y = numpy.arange(0, x)

    # to subtract height from ground
    #z = z - 609.6
    #z = z - 0.5334
    #z = z - 0.6096
    #z = z * 100
    #z = z - 61
    #z = z - 62
    #z= z - 12
    # plot 3d graph x, y and depth data
    #z[z<-30]=0
    X, Y = numpy.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pyplot.xlabel('X pixels')
    pyplot.ylabel('Y pixels')
    ax.set_zlabel('Z: Depth values(mm)')
    #pyplot.show()
    fig.savefig("dorg130") #save the image
    #numpy.savetxt("testremove.txt", z, delimiter=" ", fmt = '%.4f')
    # combine x, y and z into a 3 column matrix
    #rows = 285200
    rows = x * y
    columns = 3
    maxlength = y
    #maxlength = 620
    maxwidth = x
    #maxwidth = 460
    B = numpy.empty((rows, columns))
    row = 0
    for width in range(maxwidth):
    	for length in range(maxlength):
    		B[row, 0] = width
    		B[row, 1] = length
    		B[row, 2] = z[width, length]
    		row = row + 1
    
    # run ransac on dataset B (1st trial)
    #model_robust, inliers = ransac(B, LineModelND, min_samples=3, residual_threshold=1, max_trials=1000)
    # get the inverse of inliers
    #outliers = inliers == False
    
    return(x,y,z,B);