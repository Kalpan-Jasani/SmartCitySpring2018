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

from leastSqCoeff import leastSqCoeff
from severity import severity
#from setFigure import setFigure
from plotFigure import plotFigure


#something new
# generate x, y and z coordinates
#X = numpy.arange(0, 620) #512
#Y = numpy.arange(0, 460) #424
#z = numpy.loadtxt("24in_RealCrack1.txt", delimiter=" ")
#z = numpy.loadtxt("frame4190.txt", delimiter=" ")

z = numpy.loadtxt("data.txt", delimiter="\t") #load into 2D array

# to remove the edges
#from row 7 to end - 7 and from column 7 to end - 7
z = z[10:-10, 10:-10]
#z = z[:,420:-115]

# No hardcoding
x,y = z.shape  #gives dimensions

X = numpy.arange(y) #gives numbers from 0 to y(stored in X)
Y = numpy.arange(x) #gives numbers from 0 to x(stored in Y)

# to subtract height from ground
#z = z - 609.6
#z = z - 0.5334
#z = z - 0.6096
#z = z * 100
#z = z - 61
#z = z - 62
#z= z - 12
#plot 3d graph x, y and depth data
#z[z<-30]=0
X, Y = numpy.meshgrid(X, Y) #makes 2D array for plotting DONT WORRY ABOUT X

# remove next comments to show figure 1
fig = plotFigure(X,Y,z,z,True) #plot original figure(before adjusting)
fig.savefig("raw") #save the image

pyplot.close()



#the next few line makes 3D plot (stored in B finally) of the x,y values and their corresponding depth
rows = x * y #this stores the number of pixels.. each pixel's data is stored in B
columns = 3

maxlength = y
maxwidth = x
B = numpy.empty((rows, columns)) #initializes array B with random crap in it of dimensions (rows, columns)
row = 0
for width in range(maxwidth): 
    for length in range(maxlength):
    		B[row, 0] = width
    		B[row, 1] = length
    		B[row, 2] = z[width, length] #matches each x and y with corresponding depth
    		row = row + 1




bbox = numpy.array([float('Inf'),-float('Inf'),float('Inf'),-float('Inf'),float('Inf'),-float('Inf')])

#clones array B
points = numpy.array(B)


# xyz acquires the different pixels/points 

# find the pixel that is the minimum and the pixel that is the maximum

for xyz in points:
    #bbox[0] = min(bbox[0], xyz[0]) # min x
    #bbox[1] = max(bbox[1], xyz[0]) # max x
    #bbox[2] = min(bbox[2], xyz[1]) # min y
    #bbox[3] = max(bbox[3], xyz[1]) # max y
    if min(bbox[4], xyz[2]) == xyz[2]:
        bbox[4] = xyz[2]  # min z
        bbox[0] = xyz[0]
        bbox[2] = xyz[1]
        
    if max(bbox[5], xyz[2]) == xyz[2]:
        bbox[5] = xyz[2]  # min z
        bbox[1] = xyz[0]
        bbox[3] = xyz[1]
    #bbox[5] = max(bbox[5], xyz[2]) # max z

#finds coordinates of the center in the cube
bbox_center = numpy.array([(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2]);

#hardcodes the tolerance and threshold 
TOLERANCE = 1

#TOLERANCE is distance you allow for one point to be away from the fitted plane
# ratio of inliers

# threshold should be high. That is the ratio of inliers you want in your ransac. For a given test, you will need to have this or higher ratio to count as valid candidate set of ponts
THRESHOLD = 0.2
N_ITERATIONS = 1000
# Finds least squares solution coeffiecients for ax+by+cz=1

#this code is faulty becaue it picks points randomly from the points array. 


# write code where you store all the points not in the pothole.



road_points = numpy.empty((rows, columns))
counter = 0
for point in points:
    if point[2] < 790 and point[2] > 766:
        # paste the road pixel in to the new array
        road_points[counter, 0] = point[0]
        road_points[counter, 1] = point[1]
        road_points[counter, 2] = point[2]
        counter = counter + 1


#counter now contains the number of the picked pixels

road_points_2 = road_points[:counter,]  

     
# write code that choose, say 1000, points from the set above . IMPORTANT: These points should be chosen RANDOMLY(not the first 1000)
# give this data instead of "points" in the function leastSqCoeff

(a,b,c) = leastSqCoeff(THRESHOLD, TOLERANCE, N_ITERATIONS, road_points_2, bbox)


# Linear plane eq aX + bY + cZ = 1
Z = (1 - a*X - b*Y)/c 


fig = plotFigure(X,Y,Z,z,True)
fig.savefig("plane_test") #save as png
pyplot.close()



# Depth image subtracted from fitted plane
#depthdiff = Z  - z
#This one works
depthdiff = abs(z - Z)

# Set to 0 depth diff less than 15mm
depthdiff[depthdiff < 5] = 0

# Plot test image of both the plane and the subtracted depth data
fig = plotFigure(X,Y,Z,depthdiff,False)
fig.savefig("after_plane_fitting")

pyplot.close()

#not plotting for debugging
#pyplot.show()




#<<<<<<Need to comment everything below>>>>>>>


#Find the max depth in the array 
deepest = max(depthdiff.flatten())

#divide by deepest - normalized
test = depthdiff/ deepest # why are the values not 0 to 1?

#show grayscale img. But, not on the GUI, it "shows" on the axes (the canvas). The canvas isn't showed to GUI (at this stage)
pyplot.imshow(test)

#not plotting for debugginh
#pyplot.show()

#closing so pyplot.show in next graph doesn't open this too
pyplot.close()

pyplot.imsave("colored_2d.png", test)

#Need to adjust what i change to 0 because currently it only shows the left part
from skimage import filters

otsuimg = filters.threshold_otsu(test)
pyplot.imshow(test > otsuimg, cmap='gray', interpolation='nearest')

#not displaying for getting faster output
# pyplot.show()

xlim, ylim = pyplot.xlim(), pyplot.ylim()
pyplot.plot(x,y,"o")
pyplot.xlim(xlim)
pyplot.ylim(ylim)
pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

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

pyplot.close()

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
