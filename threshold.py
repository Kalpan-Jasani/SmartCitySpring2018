#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:02:16 2018

@author: etan
"""
import numpy
import random
import math

def threshold (THRESHOLD, TOLERANCE, N_ITERATIONS, points, bbox):
    #run ransac on dataset B (2nd trial)
    #code taken from https://github.com/minghuam/point-visualizer/blob/master/point_visualizer.py
    #http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
    # tolerance for distance, e.g. 0.0027m for kinect
    #TOLERANCE = 0.78 #5 #removing hard coding
    # ratio of inliers
    #THRESHOLD = 0.05 #removing hard coding
    N_ITERATIONS = 1000
    iterations = 0
    solved = 0
    while iterations < N_ITERATIONS and solved == 0:
        iterations += 1
        #max_error = -float('inf')
        #max_index = -1
        # randomly pick three non-colinear points
        CP = numpy.array([0,0,0])
        while CP[0] == 0 and CP[1] == 0 and CP[2] == 0:
            [A,B,C] = points[random.sample(range(len(points)), 3)]
            # make sure they are non-collinear
            CP = numpy.cross(A-B, B-C)
            # calculate plane coefficients
            abc = numpy.dot(numpy.linalg.inv(numpy.array([A,B,C])), numpy.ones([3,1]))
            # get distances from the plane
            d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])
            dist = abs((numpy.dot(points, abc) - 1)/d)
            #print max(dist),min(dist)
            ind = numpy.where(dist < TOLERANCE)[0]
            ratio = float(len(ind))/len(points)
            if ratio > THRESHOLD:
                # satisfied, now fit model with the inliers
                # least squares reference plane: ax+by+cz=1
                inliers = numpy.take(points, ind, 0)
                print('\niterations: {0}, ratio: {1}, {2}/{3}'.format(iterations, ratio,len(points),len(inliers)))
                [a,b,c] = numpy.dot(numpy.linalg.pinv(inliers), numpy.ones([len(inliers), 1]))
                plane_pts = numpy.array([
                        [bbox[0], bbox[2], (1-a*bbox[0]-b*bbox[2])/c],
                        [bbox[0], bbox[3], (1-a*bbox[0]-b*bbox[3])/c],
                        [bbox[1], bbox[3], (1-a*bbox[1]-b*bbox[3])/c],
                        [bbox[1], bbox[2], (1-a*bbox[1]-b*bbox[2])/c]])
                print('Least squares solution coeffiecients for ax+by+cz=1')
                print (a,b,c)
                solved = 1
    return (a,b,c);
