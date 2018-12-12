#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:04:21 2018

@author: rz2333
"""
import cv2
import os
import numpy as np


def thickLineImg(in_img, num_erosion, num_dialation, num_iter):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if num_dialation < 0:
        num_dialation = 0
    if num_erosion < 0:
        num_erosion = 0
    if num_iter <0:
        num_iter = 1
    
    out_img = in_img
    for i in range(num_iter):
        
        out_img = cv2.dilate(out_img, kernel, iterations=num_dialation)
        out_img = cv2.erode(out_img, kernel, iterations=num_erosion)

    return out_img

def wall_contour(in_img, approximate=True, epsil=0.1):
    #_, out_img = cv2.threshold(in_img, 177, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(in_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #cv2.approxPolyDP()
    if approximate:
        for i in range(len(contours)):
            epsilon = epsil*cv2.arcLength(contours[i], True)
            contours[i] = cv2.approxPolyDP(contours[i],epsilon,True)
    return contours, hierarchy

def distance(point1, point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**(1/2)
    
def gap_connect( corner, cutoff, big_cutoff, small_cutoff, wall_thick,img_drawing,vertical=True ):
    #num_point = corner.shape[0]
    num_point = len(corner)
    #corner = corner.astype('uint8')

    for i in range(num_point):
        point1 = corner[i]
        hori_thick = []
        vert_thick = []
        #print('Point1 is ' + str(point1))
        
        if vertical:
            #print('Start analyzing vertical')
            test_distance = []
            for point in corner:
                if ((point[1]-point1[1]<=10) and (not point == point1)): #and (not point == point1)
                    vert_thick.append(point)
                    #print('Find a close point')
                    
            #print(len(vert_thick))
            if len(vert_thick) ==0:
                vert_thick.append(point1)
            
            for point in vert_thick:
                dis = distance(point,point1)
                test_distance.append(dis)
             
            filter_dis = [fil_dis for fil_dis in test_distance if fil_dis > 8]    
            #print(len(test_distance))
            if len(filter_dis)==0:
                filter_dis.append(0)
            test_thick_point = vert_thick[test_distance.index(min(filter_dis))]
            thick = abs(point1[0] - test_thick_point[0])
            #print(point1)
            #print(test_thick_point)
            #print(thick)
    
        else:
            test_distance = []
            for point in corner:
                if (point[0]-point1[0]<=10) and (not point == point1):
                    hori_thick.append(point)
                    
            #print(len(hori_thick))
            if len(hori_thick) ==0:
                hori_thick.append(point1)
                
            for point in hori_thick:
                dis = distance(point,point1)
                test_distance.append(dis)

            filter_dis = [fil_dis for fil_dis in test_distance if fil_dis > 8]    
            if len(filter_dis)==0:
                filter_dis.append(0)
                
            test_thick_point = hori_thick[test_distance.index(min(filter_dis))]
            thick = abs(point1[1] - test_thick_point[1])
            #print(point1)
            #print(test_thick_point)
            #print(thick)
            
        for j in range(i+1, num_point):
            #point1 = corner[i]
            point2 = corner[j]
            point_distance = ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**(1/2)
            if vertical:
                if thick >wall_thick:
                    dis_cutoff = big_cutoff
                else:
                    dis_cutoff = small_cutoff
                    
                if (abs(point1[0] - point2[0])) < cutoff and (point_distance < dis_cutoff):
                    cv2.line(img_drawing,point1,point2,(255,255,255),5)
            else:
                if thick > wall_thick:
                    dis_cutoff = big_cutoff
                else:
                    dis_cutoff = small_cutoff
                    
                if (abs(point1[1] - point2[1])) < cutoff and (point_distance < dis_cutoff):
                    cv2.line(img_drawing,point1,point2,(255,255,255),5)                
                
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a



path = '/Users/rz2333/Downloads/Study/BU/Fall_2018/CS542_ml/Final_project/Data/Equation_pred/cv_process'
allfile = os.listdir(path)
cvpath = '/Users/rz2333/Downloads/Study/BU/Fall_2018/CS542_ml/Final_project/Data/Equation_gapclose'

for file in allfile:
    print('Process image ' + str(file))
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    _, img_thread = cv2.threshold(img, 177, 255, cv2.THRESH_BINARY)
    #imgthick = img_thread
    imgthick = thickLineImg(img_thread, num_erosion=1, num_dialation=2, num_iter=3)
    
    img_coutour, img_hierarchy = wall_contour(imgthick, approximate=True, epsil=0.0005)
    drawing = np.zeros((imgthick.shape[0], imgthick.shape[1], 3), np.uint8)     
    # draw contours
    for i in range(len(img_coutour)):
        color_contours = (255, 255, 255) # green - color for contours
        #color = (255, 0, 0) # blue - color for convex hull
        if (cv2.contourArea(img_coutour[i]) >= 50):  #and (cv2.contourArea(img_coutour[i]) <= 1000000)
            img_show=cv2.drawContours(drawing, img_coutour, i, color_contours, 3, 8, img_hierarchy)
            #img_show=cv2.drawContours(drawing, img_coutour, i, color_contours, -1)

    contour_corner = np.zeros((1,2))
    for i in range(len(img_coutour)):
        points = img_coutour[i]
        num, _, dim = points.shape
        points = points.reshape((num, dim))
        contour_corner = np.concatenate((contour_corner, points), axis=0)
    
    contour_corner = totuple(contour_corner.astype('int16'))

    gap_connect(contour_corner, cutoff=10, big_cutoff=500, small_cutoff=200,wall_thick=20,img_drawing=drawing, vertical=True) 
    gap_connect(contour_corner, cutoff=10, big_cutoff=700, small_cutoff=200,wall_thick=20,img_drawing=drawing, vertical=False) 
     
    cv2.imwrite(os.path.join(cvpath, file), drawing)  