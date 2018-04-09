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
    
    cross_product_vector = numpy.array([0,0,0])
    [a, b, c] = numpy.array([0, 0, 0])      # declaring this dude ( or these dudes)
    
    while iterations < N_ITERATIONS:
        iterations += 1
        
        # pick three random points from points
        [A,B,C] = points[random.sample(range(len(points)), 3)]
        
        
        # to make sure they are non-collinear, we make a cross product and check if that vector is 0, 0, 0
        
        #this line gets a normal vector
        cross_product_vector = numpy.cross(A-B, B-C)
    
        if cross_product_vector[0] == 0 and cross_product_vector[1] == 0 and cross_product_vector[2] == 0:
            continue
        # calculate plane coefficients
        
        
        # solves ax + by + cz = 1, gets a, b, and c values
        # Note: that is the equation of a plane (and not THE PLANE), we will test number of outliers and ratio and stuff with this plane
        abc = numpy.dot(numpy.linalg.inv(numpy.array([A,B,C])), numpy.ones((3,1)))
       
        
        # get length of normal vector
        # normal vector is vector perpendicular to the plane
        
        len_normal_vector = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])
        
        #get distances from the plane IMPORTANT STEP
        # in this case numpy.dot is same as matrix multiplication.
        # all in all, dist is having many rows (as many as those in points), and just 1 column. dist contains distance of each points from the plane
        dist = abs((numpy.dot(points, abc) - 1)/ len_normal_vector)
        
        # for reference of why formula above is correct: https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/dot-cross-products/v/point-distance-to-plane
        
        
        # ind is an ndarray (numpy array) containing indices of all road points :)
        ind = numpy.where(dist < TOLERANCE)[0]
    
        ratio = float(len(ind))/len(points)
        if ratio > THRESHOLD:   # good plane, as there are enough inliers
            if ratio > highest_ratio:   # if previously we had a better model, skip this one
                inliers = numpy.take(points, ind, 0)
                
                # least squares reference plane: ax+by+cz=1
                # reference: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
                [a,b,c] = numpy.dot(numpy.linalg.pinv(inliers), numpy.ones((len(inliers), 1)))
                
                # the above attempts to solve the equation of the plane, given all the inlier points (this is done with Ax = b kind of stuff in linear algebra)
                
                highest_ratio = ratio #set the current ration as the highest ratio, as it is, uptill now, the highest ratio ever encountered.
    return (a,b,c)  #returning a tuple (which is kind of like a simple list)
