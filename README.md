# CS542-SpaceXYZ

This is a project for CS542 BU. <br>

Project Name: Space XYZ <br>
Team members: Rui Hong, Jingxuan Wu, Wenjun Zhu, Shichen Cao

1. Goal

The project is aimming to automatically calculate the area of each room in a floor plan

2. Procedures

(1). OCR using EAST with openCV and pytesseract in the original image to identify room type/name (Wenjun Zhu)

(2). apply FCN(pytorch) and Unet(tf keras) to do the room segmentation (Rui Hong, Wenjun Zhu)

(3). image preprocess with image morphological operations with openCV (Wenjun Zhu)

(4). close the gaps(doors, windows) after preprocessing (Rui Hong)

(5). apply connected component algorithm with opencv to seperate each room and count labels 

(6). use SSD(tf) to detect the scale bar (Jingxuan Wu, Shichen Cao)

(7). use scale bar number / pixel in the detection zone to compute the area 

