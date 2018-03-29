#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:02:16 2018

@author: ethan
"""
import numpy
import random
import math


#THRESHOLD is the ratio you want to achieve, proportion of inliers in the total set, for a given test. If ot hits, test succeds and iterations stop
# TOLERANCE is the value that says if a point is a valid inlier

# TODO: check if points can be used (shadowing of global variable)

def leastSqCoeff (THRESHOLD, TOLERANCE, N_ITERATIONS, points, bbox):
    #points is a roster or a list of all the points
    
    #run ransac on dataset B (2nd trial)
    #code taken from https://github.com/minghuam/point-visualizer/blob/master/point_visualizer.py
    #http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
    
    iterations = 0
    highest_ratio = 0
    
    CP = numpy.array([0,0,0])
    [a, b, c] = numpy.array([0, 0, 0])      # declaring this dude ( or these dudes)
    
    while iterations < N_ITERATIONS:
        iterations += 1
        
        
        
        [A,B,C] = points[random.sample(range(len(points)), 3)]
        
        
        # make sure they are non-collinear
        
        #this line gets a normal vector
        CP = numpy.cross(A-B, B-C)
    
        if CP[0] == 0 and CP[1] == 0 and CP[2] == 0:
            continue
        # calculate plane coefficients
        
        
        # solves ax + by + cz = 1, gets a, b, and c values
        abc = numpy.dot(numpy.linalg.inv(numpy.array([A,B,C])), numpy.ones((3,1)))
       
        
        # get length of normal vector
        len_normal_vector = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])
        
        #get distances from the plane IMPORTANT STEP
        dist = abs((numpy.dot(points, abc) - 1)/ len_normal_vector)
        
        
        # ind is an ndarray (numpy array) containing indices of all road points :)
        ind = numpy.where(dist < TOLERANCE)[0]
    
        ratio = float(len(ind))/len(points)
        if ratio > THRESHOLD:
            # satisfied, now fit model with the inliers
            # least squares reference plane: ax+by+cz=1
            if ratio > highest_ratio:
                inliers = numpy.take(points, ind, 0)
                [a,b,c] = numpy.dot(numpy.linalg.pinv(inliers), numpy.ones((len(inliers), 1)))
                highest_ratio = ratio
    return (a,b,c);
