from imutils.object_detection import non_max_suppression
import pytesseract
import cv2
import numpy as np
import os
import re
import io
import pandas as pd

building = ['oxygen', 'eko', 'equation']
gapclose = ['./Oxygen_gapclose/', './Eko_gapclose/', './Equation_gapclose/']
csvfiles = ['bondingboxes_oxygen.csv', 'bondingboxes_eko.csv', 'bondingboxes_equation.csv']

imgnum = -1

# ref: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# use east model to do predictions
# param:
#    imgfile: input image path
#    model: east model path
#    targetH, targetW: resize width and height
# return:
#    boundary: the coordinates of upper left and bottom right points
#    probs: probability of each offset point being a text or the score of box arround that pixel
#    rH, rW: ratio of original height and width over target height and width
def textDetection(imgfile, model, targetH, targetW):
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    H, W = img.shape[:-1]
    rH, rW = float(H) / targetH, float(W) / targetW
    img = cv2.resize(img, (targetW, targetH))

    layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
    east = cv2.dnn.readNet(model)
    # RGB mean to reduce the influence of illumination, mean is the same as GoogLeNet
    blob = cv2.dnn.blobFromImage(img, 1.0, (targetW, targetH), (0, 0, 0), swapRB=True, crop=False)
    east.setInput(blob)
    scores, geometry = east.forward(layer_names)

    rows, cols = scores.shape[2:4]
    boundary = []
    probs = []

    for r in range(rows):
        for c in range(cols):
            up, right, down, left, angle = geometry[0, :, r, c]
            prob = scores[0][0][r][c]
            offsetX, offsetY = c * 4, r * 4
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = up + down
            w = left + right

            endX = int(offsetX + cos * right + sin * down)
            endY = int(offsetY + cos * down - sin * right)
            startX = max(endX - w, 0)
            startY = max(endY - h, 0)

            if endX - startX > 50: continue
            if endY - startY > 15: continue

            boundary.append((startX, startY, endX, endY))
            probs.append(prob)
    return boundary, probs, (rH, rW)

'''
getRoomcentroid use the result from text detection to do ocr with tesseract to get the
centroids of each type of room
param:
    boundary: diagonal points of a bounding box
    probs: probability of each box
    rH, rW: height and width ratio
    imgpath: original image path
return:
    the centroids of chambre, cuisine, wc, sdb
'''
def getRoomCentroid(boundary, probs, rH, rW, imgpath):
    global imgnum
    imgnum += 1
    # same order as pattern
    room_centroid = [[], [], [], []]

    pattern = [r'.*chambre.*', r'.*cuisine.*', r'.*wc.*', r'.*sdb.*']

    boxes = non_max_suppression(np.array(boundary), probs=probs)

    origin_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

    for startX, startY, endX, endY in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        img = origin_img[startY: endY+1, startX: endX+1, :]

        text = pytesseract.image_to_string(img).lower()

        alphanum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ &0123456789"
        if any([a in text for a in alphanum]):
            cv2.rectangle(origin_img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        for i, p in enumerate(pattern):
            if re.match(p, text):
                centroid = [(startY + endY) // 2, (startX + endX) // 2]
                room_centroid[i].append(centroid)

    cv2.imwrite("text_detection{}.jpg".format(imgnum), origin_img)

    return room_centroid[0], room_centroid[1], room_centroid[2], room_centroid[3]

def seedFlooding(img, row_c, col_c, label):
    assert img.shape[2] == 1, print("must be a binary image")

    r_max, c_max = img.shape[0], img.shape[1]

    if row_c < 0 or row_c >= r_max or col_c < 0 or col_c >= c_max:
        return

    if img[row_c][col_c] == 0 or img[row_c][col_c] == label:
        return

    img[row_c][col_c] = label

    # left
    seedFlooding(img, row_c-1, col_c, label)
    # up
    seedFlooding(img, row_c, col_c-1, label)
    # right
    seedFlooding(img, row_c, col_c+1, label)
    # down
    seedFlooding(img, row_c+1, col_c, label)

    return


'''
This function do room detection using connected component algorithm based on the ocr result
param:
    chambre: centroids for chambre
    cuisine: centroids for cuisine
    wc: centroids for wc
    sdb: centroids for sdb
    img: the preprocessed image
    scale: area per pixel estimated
return:
    ret_img: return color image with colored room and area showed in the room
    area_chambre: area of each chambre(bedroom)
    area_cuisine: area of each cuisine(living room and kitchen)
    area_wc: area of wc(toilet)
    area_sdb: area of sdb(bathroom)
    wcsdb: area of bathroom and toilet
    room area orders are the same as the centroids orders
'''
def room_detection(chambre, cuisine, wc, sdb, img, scale):
    connectivity = 4

    n_labels, label_mat, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=connectivity)

    ret_img = np.ones((label_mat.shape[0], label_mat.shape[1], 3), dtype=np.uint8)*255

    back_label = np.argmax(stats[:, -1])

    #chambre: purple
    purple = []
    area_chambre = []
    #cuisine: red
    red = []
    area_cuisine = []
    #wc: green
    green = []
    area_wc = []
    #sdb: blue
    blue = []
    area_sdb = []
    #wc + sdb: yellow
    yellow = []
    wcsdb = []

    for centroid in chambre:
        purple.append(label_mat[int(centroid[1]), int(centroid[0])])
        area_chambre.append(stats[label_mat[int(centroid[1]), int(centroid[0])], -1])

    for centroid in cuisine:
        red.append(label_mat[int(centroid[1]), int(centroid[0])])
        area_cuisine.append(stats[label_mat[int(centroid[1]), int(centroid[0])], -1])

    for centroid in wc:
        green.append(label_mat[int(centroid[1]), int(centroid[0])])
        area_wc.append(stats[label_mat[int(centroid[1]), int(centroid[0])], -1])

    for centroid in sdb:
        if label_mat[int(centroid[1]), int(centroid[0])] in green:
            green.remove(label_mat[int(centroid[1]), int(centroid[0])])
            yellow.append(label_mat[int(centroid[1]), int(centroid[0])])
            area_wc.remove(stats[label_mat[int(centroid[1]), int(centroid[0])], -1])
            wcsdb.append(stats[label_mat[int(centroid[1]), int(centroid[0])], -1])
        else:
            blue.append(label_mat[int(centroid[1]), int(centroid[0])])
            area_sdb.append(stats[label_mat[int(centroid[1]), int(centroid[0])], -1])

    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            if label_mat[i, j] == 0:
                ret_img[i, j, :] = 0
            elif label_mat[i, j] != back_label and label_mat[i, j] in purple:
                ret_img[i, j, :] = [255, 100, 255]
            elif label_mat[i, j] != back_label and label_mat[i, j] in red:
                ret_img[i, j, :] = [100, 100, 255]
            elif label_mat[i, j] != back_label and label_mat[i, j] in green:
                ret_img[i, j, :] = [100, 255, 100]
            elif label_mat[i, j] != back_label and label_mat[i, j] in blue:
                ret_img[i, j, :] = [255, 100, 100]
            elif label_mat[i, j] != back_label and label_mat[i, j] in yellow:
                ret_img[i, j, :] = [100, 255, 255]

    for i in range(len(chambre)):
        cv2.putText(ret_img, str(int(area_chambre[i]*scale)) + 'm^2', (chambre[i][0] - 100, chambre[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=5)
    for i in range(len(cuisine)):
        cv2.putText(ret_img, str(int(area_cuisine[i]*scale)) + 'm^2', (cuisine[i][0] - 100, cuisine[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=5)
    for i in range(len(wc)):
        if len(area_wc) == 0:
            cv2.putText(ret_img, str(int(scale*wcsdb[i])) + 'm^2', (wc[i][0] - 100, wc[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=(0, 0, 0), thickness=5)
        else:
            cv2.putText(ret_img, str(int(scale*area_wc[i])) + 'm^2', (wc[i][0] - 100, wc[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=(0, 0, 0), thickness=5)
    for i in range(len(sdb)):
        if len(area_sdb) == 0:
            cv2.putText(ret_img, str(int(scale*wcsdb[i])) + 'm^2', (wc[i][0] - 100, wc[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=(0, 0, 0), thickness=5)
        else:
            cv2.putText(ret_img, str(int(scale*area_sdb[i])) + 'm^2', (sdb[i][0] - 100, sdb[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=(0, 0, 0), thickness=5)

    area_chambre = list(np.array(area_chambre) * scale)
    area_cuisine = list(np.array(area_cuisine) * scale)
    area_wc = list(np.array(area_wc) * scale)
    area_sdb = list(np.array(area_sdb) * scale)
    wcsdb = list(np.array(wcsdb) * scale)

    return ret_img, area_chambre, area_cuisine, area_wc, area_sdb, wcsdb


'''
This function is used to remove small components created by preprocessing
param:
    img: an binary image being processed, should be inversed(background = white, object = black)
    num_pixel: threshold pixel, components have pixels less than this threshold will be dropped as background
'''
def clean_img(img, num_pixel):
    connectivity = 4

    n_labels, label_mat, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=connectivity)

    ret_img = np.ones_like(label_mat)*255

    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            if stats[label_mat[i, j], -1] > num_pixel and label_mat[i, j] != 0:
                ret_img[i, j] = 0

    ret_img = ret_img.astype(np.uint8)

    #cv2.imshow('win', ret_img)
    #cv2.waitKey(60)

    return ret_img


'''
This function is used to do preprocessing, removing auxiliary lines, numbers and shadow of each images
Keep using image erosion and dilation to remove noises
param:
    path: input images folder path
    w_path: write path. Folder of the output images
'''
def get_wall_img(path: str, w_path: str) -> None:

    imgs = os.listdir(path)

    for filename in imgs:
        name = path + filename
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        _, img_thread = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)

        img_clean = clean_img(img_thread, 1000)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        walls = cv2.dilate(img_clean, kernel, iterations=2)
        _, nwalls = cv2.threshold(walls, 250, 255, cv2.THRESH_BINARY_INV)
        nwalls = clean_img(nwalls, 800)
        nwalls = cv2.erode(nwalls, kernel, iterations=10)
        nwalls = cv2.dilate(nwalls, kernel, iterations=13)
        _, nwalls = cv2.threshold(nwalls, 250, 255, cv2.THRESH_BINARY_INV)
        nwalls = clean_img(nwalls, 800)
        nwalls = cv2.erode(nwalls, kernel, iterations=5)
        nwalls = cv2.morphologyEx(nwalls, cv2.MORPH_OPEN, kernel, iterations=30)
        _, nwalls = cv2.threshold(nwalls, 250, 255, cv2.THRESH_BINARY_INV)
        nwalls = clean_img(nwalls, 3000)

        cv2.imwrite(w_path + filename, nwalls)

    print('finished')


def main():
    for i, build in enumerate(building):
        if not os.path.exists('./CV_{}/'.format(build.upper())):
            os.makedirs('./CV_{}'.format(build.upper()))
            path = './{}/'.format(build)
            get_wall_img(path, './CV_{}/'.format(build.upper()))
        else:
            df = pd.read_csv(csvfiles[i])
            w_path = './' + build + '_result/'
            if os.path.exists(gapclose[i]):
                for file in os.listdir(gapclose[i]):
                    img = cv2.imread(gapclose[i] + file, cv2.IMREAD_GRAYSCALE)
                    _, img_b = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
                    boundary, probs, ratio = textDetection('./' + build + '/' + file,
                                                           './frozen_east_text_detection.pb', 320, 320)
                    chambre, cuisine, wc, sdb = getRoomCentroid(boundary, probs, ratio[0], ratio[1],
                                                                './' + build + '/' + file)
                    scale = df[df['filename'].isin([file])]['m2perpixel']
                    img_seg, a_chambre, a_cuisine, a_wc, a_sdb, wcsdb = room_detection(chambre, cuisine, wc, sdb,
                                                                                       img_b, scale)
                    cv2.imwrite(w_path + file, img_seg)


if __name__ == "__main__":
    main()
