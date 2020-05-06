# CS542-SpaceXYZ

This is a project for CS542 BU. <br>

Project Name: Space XYZ <br>
Team members: Rui Hong, Jingxuan Wu, Wenjun Zhu, Shichen Cao

1. Goal

The project is aimming to automatically calculate the area of each room in a floor plan

2. Procedures

(1). image preprocessing with image morphological operations with opencv

(2). try to apply FCN to do the room segmentation with pytorch

(3). closing the gaps(doors, windows) after preprocessing

(4). connected component with opencv

(5). OCR using google api in the original image to identify each room

(6). use SSD to detect the scale bar

(7). use scale bar number / pixel in the detection zone to compute the area

