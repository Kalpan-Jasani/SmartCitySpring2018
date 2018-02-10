import cv2
import numpy
import matplotlib.pyplot as pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac
import scipy.linalg
import random
import math
import time 

from threshold import threshold
from severity import severity
from setFigure import setFigure

#this function sets up the x and y axes to print the graphics on
(X,x,Y,y,z,B) = setFigure()

# prepare the B dataset for trial 2
point_list = []
bbox = numpy.array([float('Inf'),-float('Inf'),float('Inf'),-float('Inf'),float('Inf'),-float('Inf')])

points = numpy.array(B)


for xyz in points:
    bbox[0] = min(bbox[0], xyz[0]) # min x
    bbox[1] = max(bbox[1], xyz[0]) # max y
    bbox[2] = min(bbox[2], xyz[1]) # min x
    bbox[3] = max(bbox[3], xyz[1]) # max y
    bbox[4] = min(bbox[4], xyz[2]) # min z
    bbox[5] = max(bbox[5], xyz[2]) # max z

bbox_corners = numpy.array([
    [bbox[0],bbox[2], bbox[4]],
    [bbox[0],bbox[2], bbox[5]],
    [bbox[0],bbox[3], bbox[5]],
    [bbox[0],bbox[3], bbox[4]],
    [bbox[1],bbox[3], bbox[4]],
    [bbox[1],bbox[2], bbox[4]],
    [bbox[1],bbox[2], bbox[5]],
    [bbox[1],bbox[3], bbox[5]]]);

bbox_center = numpy.array([(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2]);

#run ransac on dataset B (2nd trial)
#code taken from https://github.com/minghuam/point-visualizer/blob/master/point_visualizer.py
#http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
# tolerance for distance, e.g. 0.0027m for kinect

TOLERANCE = 0.78 #5
# ratio of inliers
THRESHOLD = 0.05
N_ITERATIONS = 1000
# Finds least squares solution coeffiecients for ax+by+cz=1
(a,b,c) = threshold(THRESHOLD, TOLERANCE, N_ITERATIONS, points, bbox)

# plot ransac
#fig = pyplot.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(points[ind][:,0], points[ind][:,1], points[ind][:,2], c='b', marker='o', label='Inlier data')
#fig.savefig("dorg_130ransacInliers") #save the image
#ax.scatter(points[outliers][:,0], points[outliers][:,1], points[outliers][:,2], c='b', marker='o', label='Outlier data')
#pyplot.show()
#fig.savefig("dorg_19ransacOutliers") #save the image

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

# Depth image subtracted from fitted plane
#depthdiff = Z  - z
#This one works
depthdiff = z - Z 
# Set to 0 depth diff greater than 5mm
depthdiff[depthdiff > 5] = 0
depthdiff[depthdiff > 0] = 0
#this one works
#mask = (depthdiff > 0) & (depthdiff < 25)
#mask = (depthdiff > 0) and (depthdiff < 5)
#depthdiff[mask] = 0

# Plot test image of both the plane and the subtracted depth data
fig = pyplot.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,depthdiff, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)
#pyplot.show()
pyplot.xlabel('X pixels')
pyplot.ylabel('Y pixels')
ax.set_zlabel('Z: Depth values(mm)')
fig.savefig("dorg26plots")
#fig.savefig("dorg26plots")
pyplot.show()
# trial 2 because 5mm seems to still have a lot of points above the plane
# this time set to 2mm
#depthdiff[depthdiff > 5] = 0
#depthdiff[depth > 0] = 0
#depthdiff[depthdiff > -5] = 0

#Find the max depth in the array 
deepest = min(depthdiff.flatten())
#divide by deepest - normalized
test = depthdiff/ deepest # why are the values not 0 to 1?
#show grayscale img
pyplot.imshow(test)
pyplot.show()
#pyplot.imsave("depthdiff19.png", test)

#Need to adjust what i change to 0 because currently it only shows the left part
from skimage import filters
otsuimg = filters.threshold_otsu(test)
pyplot.imshow(test > otsuimg, cmap='gray', interpolation='nearest')
#pyplot.show()
xlim, ylim = pyplot.xlim(), pyplot.ylim()
pyplot.plot(x,y,"o")
pyplot.xlim(xlim)
pyplot.ylim(ylim)
pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
#pyplot.savefig('test')
#pyplot.show()
#pyplot.imsave("test.png", test > otsuimg, cmap='gray')


#Values positive negative problem?

#calculate percentage and occurrences of number of times it exceeds the threshold
#count = 0
#for i in range(maxwidth):
#    for j in range(maxlength):
#        if test[i][j] > otsuimg:
#            count = count + 1
#count


# clear plot
#pyplot.clf()

#save the image
pyplot.axis('off')
pyplot.savefig('test.png', bbox_inches='tight', pad_inches=0)

#read in the image
img = cv2.imread('test.png', 0)
img3 = cv2.imread('otsu19.png', 0)
black = cv2.imread('blackbg.png', 0)

#convert to binary
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#Insted of binary, use canny edges, see if its more accurate
#edges = cv2.Canny(img, 100, 200)
#find contours
#test, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
test, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#after test drawing out the contours, correct contour is 8
cnt = contours[8]
# for test drawing
#cv2.drawContours(img2, contours, -1, (255,255,0), 3)
#pyplot.imshow(img2)
#pyplot.show()
#find bounding box
rect = cv2.minAreaRect(cnt)
#rect (center(x,y),(width, height), angle of rotation)
box = cv2.boxPoints(rect)
box = numpy.int0(box)
#cv2.drawContours(img3, [box], 0, (255,255,0),3)

#pyplot.imshow(img3)
#ax.set_title('X length %.2f', rect[1][0])
#pyplot.xlabel('X pixels')
#pyplot.ylabel('Y pixels')
#pyplot.show()

#Calculations!!!!
#find the area of the contour?
#area = cv2.contourArea(cnt)
#rect[1]
horizontalField = 57 #angle degree
verticalField = 43 #degree
height = 778 #mm
widthRes = 640 #pixel
lengthRes = 480 #pixel
kinectAngle = math.radians(horizontalField) #convert angle from degree to radians
lengthAct = 2*height*math.tan(kinectAngle/2)
pixelLength =  lengthAct/widthRes
widthdiameter = rect[1][0]
lengthdiameter = rect[1][1]
widthdiameter = widthdiameter * pixelLength
lengthdiameter = lengthdiameter * pixelLength
avgDiameter = (widthdiameter+lengthdiameter)/2

#if else statements for severity level using @avgDiameter and @deepest
deepest = abs(deepest)
#severity function is in another file, it prints L,M,H for low, medium, and high severity
severity(deepest, avgDiameter)
#print(time.time() - start_time)
