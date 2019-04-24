#!/usr/bin/env python
import numpy as np
import cv2
from random import randint
from math import ceil


class circle_RANSAC():

    # This class witholds all the methods that are needed for the RANSAC algorithm to work

    def __init__(self):
        self.points_matrix = None
        self.min_samp = 0
        self.threshold = 0
        self.max_ite = 0
        self.matrix_length = 0
        self.r_min = 0
        self.r_max = 0

    def check_colinearity(self,p0, p1, p2):

        #source: https://www.geeksforgeeks.org/program-check-three-points-collinear/
      
        a = p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]) 
    
        if (a == 0): 
            return 1
        else: 
            return 0

    def random_points(self, img_bin):

        # This method is responsible for selecting 3 random points from the binary distribution
        self.points_matrix = np.array(np.nonzero(img_bin)).T
        self.matrix_length = int(len(self.points_matrix))
        random_1 = randint(0, self.matrix_length)
        random_2 = randint(0, self.matrix_length)
        random_3 = randint(0, self.matrix_length)
        p0 = self.points_matrix[random_1]
        p1 = self.points_matrix[random_2]
        p2 = self.points_matrix[random_3]
        if (p0[1] == p1[1]) or (p0[1] == p2[1]) or (p1[1] == p2[1]):                        # Check if the y coordenate of the points are equal (we don't want a division by zero)
            self.random_points(img_bin)
        dp0p1 = np.sqrt(np.power((p1[0]-p0[0]), 2) + np.power((p1[1]-p0[1]), 2))
        dp0p2 = np.sqrt(np.power((p2[0]-p0[0]), 2) + np.power((p2[1]-p0[1]), 2))            
        dp1p2 = np.sqrt(np.power((p2[0]-p1[0]), 2) + np.power((p2[1]-p1[1]), 2))
        if ((dp0p1 < 10) or (dp1p2 < 10) or (dp0p2 < 10)):                      # Check if the points are too close to each other
            self.random_points(img_bin)
        x = self.check_colinearity(p0, p1, p2)
        if (x == 1):
            self.random_points(img_bin)
        return p0, p1, p2

    def ortogonal_line_points(self, p0, p1, p2):
        
        # Now we have 3 random points. We stablish two lines rp0p1 and rp1p2, and we have two other lines, one that is ortogonal to p0p1 and passes trough the middle point between p0 and p1
        # and another that is one that is ortogonal to p1p2 and passes trough the middle point between p1 and p2. The intersection of those two lines will be the possible circle centre.
        # This method generates 4 points. Two of them belonging to the ortogonal line to p0p1 and tow of them beloning to the ortogonal line to p1p2

        a1 = np.zeros(2)
        a2 = np.zeros(2)
        b1 = np.zeros(2)
        b2 = np.zeros(2)

        a1[0] = (p0[0]+p1[0])/2             # a1 and a2 are two points that belong to the ortogonal line to p0p1
        a1[1] = (p0[1]+p1[1])/2

        a2[0] = 1 + a1[0]
        a2[1] = a1[1] - ((p1[0]-p0[0])/(p1[1]-p0[1]))

        b1[0] = (p1[0]+p2[0])/2
        b1[1] = (p1[1]+p2[1])/2                     # b1 and b2 are two points that belong to the ortogonal line to p1p2
                                                                    
        b2[0] = 1 + b1[0]
        b2[1] = b1[1] - ((p2[0]-p1[0])/(p2[1]-p1[1]))

        return a1, a2, b1, b2

    def find_centre(self, a1, a2, b1, b2):

        # This method will calculate the intersection point coordenates between the two ortogonal lines and as well the radius of the possible circle
        # source: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

        c = np.zeros(2)
        s = np.vstack([a1,a2,b1,b2])                        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))                 # h for homogeneous
        l1 = np.cross(h[0], h[1])                           # get first line
        l2 = np.cross(h[2], h[3])                           # get second line
        x, y, z = np.cross(l1, l2)                          # point of intersection
        if z == 0:                                          # lines are parallel
            return (float('inf'), float('inf'))
        else:
            c[0] = x/z
            c[1] = y/z
            r = np.sqrt(np.power((c[0]-a1[0]), 2) + np.power((c[1]-a1[1]), 2))

        return c, r

    def points_tester(self, c, r):

        # This method is responsible for testing the other points in the binary distribution to see if they are good enough to be inliers or not

        inlier = np.zeros(shape = (self.matrix_length, 2))
        min_inliers = self.min_samp
        for i in range(self.matrix_length):
            dist_cent = np.sqrt(np.power((self.points_matrix[i][0]-c[0]), 2) + np.power((self.points_matrix[i][1]-c[1]), 2))
            d_circ = dist_cent - r                  # we calculate the distance from the point to the possible circle, if it is smaller than the threshold, it is an inlier
            if (d_circ < self.threshold):
                inlier[i] = self.points_matrix[i]
            else:
                inlier[i] = 0
        inlier = np.argwhere(inlier)
        num_inliers = len(inlier)
        if (num_inliers >= min_inliers and r > self.r_min and r < self.r_max):   # we fill the inlier tuple until the number of inliers is bigger than the minimum specified
            return c, r
        else:
            return c, 0

    def circle_ransac(self, img_bin, threshold, min_samp, max_ite, r_min, r_max):

        # This method is the RANSAC algorithm for the circle detection

        self.threshold = threshold
        self.min_samp = min_samp
        self.max_ite = max_ite
        self.r_min = r_min
        self.r_max = r_max
        i = 0
        while (i < self.max_ite):
            p0, p1, p2 = self.random_points(img_bin)
            a1, a2, b1, b2 = self.ortogonal_line_points(p0, p1, p2)
            c, r = self.find_centre(a1, a2, b1, b2)
            c, r = self.points_tester(c, r)
            if (r != 0):
                break
            else:
                i += 1
        return c, r

class image_treatment():

    # This class is responsible for getting the specified image where to look for circles and generating the binary distribution.

    def __init__(self):
        self.img_original = None
        self.img_bin = None
        cv2.namedWindow('detected_circle', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

    def img_treat(self):

        # This method will load the image and generate the binary distribution

        self.img_original = cv2.imread('/home/leleo/Documents/BIR/computer-vision/3balls.jpeg',1)
        # img_bi = cv2.bilateralFilter(self.img_original,100,150,50)
        # img_gray = cv2.cvtColor(img_bi, cv2.COLOR_BGR2GRAY)
        #img_shrinked = cv2.resize(self.img_original, (0,0), fx=0.2, fy=0.2, interpolation = cv2.INTER_LINEAR)
        img_bi = cv2.bilateralFilter(self.img_original,100,150,50)
        #img_shrinked = cv2.resize(img_bi, (0,0), fx=5, fy=5, interpolation = cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(img_bi, cv2.COLOR_BGR2GRAY)
        self.img_bin = cv2.Canny(img_gray,50,200)
        #cv2.imshow('edges',self.img_bin)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return self.img_bin

    def draw_circle(self, c, r):

        # This method is responsible for drawing the detected circle on top of the original image

        cx = ceil(c[0])
        cy = ceil(c[1])
        r = ceil(r)

        detected_circle = cv2.circle(self.img_original.copy(), (cx,cy), r, (0,255,0), 10)
        cv2.imshow('detected_circle',detected_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def main():

    # main code for the program

    it = image_treatment()          # instantiate the image_treatment class
    cR = circle_RANSAC()            # instantiate the circle_RANSAC class
    img_bin = it.img_treat()        # gets the binary distribution of the image
    h, w = img_bin.shape
    minhw = np.amin([h,w])
    r_min = np.uint16(0.1*minhw)
    r_max = np.uint16(0.2*minhw)
    c,r = cR.circle_ransac(img_bin, threshold=3, min_samp=1000, max_ite=2000, r_min=r_min, r_max=r_max)           # pass it trough the RANSAC algorithm
    print(c, r)                     # prints the circle coordinates and the radius to the terminal
    it.draw_circle(c, r)            # this shows the image with the detected circle drawn in

if __name__ == '__main__':
    main()