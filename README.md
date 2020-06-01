# CS542-SpaceXYZ

This is a project for CS542 BU. <br>

Project Name: Space XYZ <br>
Team members: Rui Hong, Jingxuan Wu, Wenjun Zhu, Shichen Cao

1. Goal

The project is aimming to automatically calculate the area of each room in a floor plan

2. Procedures

(1). image preprocessing with image morphological operations with opencv (Wenjun Zhu)

(2). applying FCN and Unet to do the room segmentation (Rui Hong, Wenjun Zhu)

(3). closing the gaps(doors, windows) after preprocessing (Rui Hong)

(4). applying connected component algorithm with opencv to seperate each room and count labels 

(5). OCR using EAST with openCV and pytesseract in the original image to identify room type/name (Wenjun Zhu)

(6). use SSD to detect the scale bar (Jingxuan Wu, Sichen Cao)

(7). use scale bar number / pixel in the detection zone to compute the area 

