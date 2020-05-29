from google.cloud import vision
import cv2
import numpy as np
import os
import re
import io
import pandas as pd

building = ['oxygen', 'eko', 'equation']
gapclose = ['./Oxygen_gapclose/', './Eko_gapclose/', './Equation_gapclose/']
csvfiles = ['bondingboxes_oxygen.csv', 'bondingboxes_eko.csv', 'bondingboxes_equation.csv']

def text_detection(img_path):
    #credit: https://cloud.google.com/vision/docs/detecting-text#vision-text-detection-python
    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as img_f:
        content = img_f.read()

    img = vision.types.Image(content=content)
    img_context = vision.types.ImageContext(language_hints=['en'])
    response = client.text_detection(image=img, image_context=img_context)

    return response


def data_process(response, im):
    chambre = []
    wc = []
    cuisine = []
    sdb = []

    contours = []

    for text in response.text_annotations:
        if re.match(r'.*chambre.*', text.description.lower()) is not None:

            a = np.array([np.zeros([1, 2], dtype=np.int32)], dtype=np.int32)

            for vertex in text.bounding_poly.vertices:
                b = np.array([np.array([vertex.x, vertex.y], dtype=np.int32).reshape((1, 2))], dtype=np.int32)
                a = np.concatenate((a, b), axis=0)
            contour = a[1:, :]
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            chambre.append([cx, cy])

            #contours.append(contour)
            #cv2.circle(im, (cx, cy), 10, (255, 0, 255), -1)

        elif re.match(r'.*s.jour.*', text.description.lower()) is not None:

            a = np.zeros([1, 2], dtype=np.int32)

            for vertex in text.bounding_poly.vertices:
                b = np.array([np.array([vertex.x, vertex.y], dtype=np.int32).reshape((1, 2))], dtype=np.int32)
                a = np.concatenate((a, b), axis=0)
            contour = a[1:, :]
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cuisine.append([cx, cy])

            #contours.append(contour)
            #cv2.circle(im, (cx, cy), 10, (0, 0, 255), -1)

        elif re.match(r'.*wc.*', text.description.lower()) is not None:

            a = np.array([np.zeros([1, 2], dtype=np.int32)], dtype=np.int32)

            for vertex in text.bounding_poly.vertices:
                b = np.array([np.array([vertex.x, vertex.y], dtype=np.int32).reshape((1, 2))], dtype=np.int32)
                a = np.concatenate((a, b), axis=0)
            contour = a[1:, :]
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            wc.append([cx, cy])

            #contours.append(contour)
            #cv2.circle(im, (cx, cy), 10, (0, 255, 0), -1)

        elif re.match(r'.*(sdb|basin).*', text.description.lower()) is not None:

            a = np.array([np.zeros([1, 2], dtype=np.int32)], dtype=np.int32)

            for vertex in text.bounding_poly.vertices:
                b = np.array([np.array([vertex.x, vertex.y], dtype=np.int32).reshape((1, 2))], dtype=np.int32)
                a = np.concatenate((a, b), axis=0)
            contour = a[1:, :]
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            sdb.append([cx, cy])

            #contours.append(contour)
            #cv2.circle(im, (cx, cy), 10, (255, 0, 0), -1)

    #cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
    #cv2.namedWindow('img_con', cv2.WINDOW_NORMAL)
    #cv2.imshow('img_con', im)
    #cv2.waitKey(30000)

    return chambre, cuisine, wc, sdb


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

    return ret_img, area_chambre, area_cuisine, area_wc, area_sdb, wcsdb


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


def get_wall_img(path, w_path):

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


def remove_color(img_c):
    for i in range(img_c.shape[0]):
        for j in range(img_c.shape[1]):
            if not (img_c[i][j][0] > 240 and img_c[i][j][1] > 240 and img_c[i][j][2] > 240):
                sum = 0
                sum1 = 0
                for k in range(i - 10, i + 11):
                    if img_c[k][j][0] < 180 and img_c[k][j][1] < 180 and img_c[k][j][2] < 180:
                        sum += 1
                for l in range(j - 10, j + 11):
                    if img_c[i][l][0] < 180 and img_c[i][l][1] < 180 and img_c[i][l][2] < 180:
                        sum += 1
                if (img_c[i][j][0] < 180 and img_c[i][j][1] < 180 and img_c[i][j][2] < 180):
                    img_c[i, j, :] = [0, 0, 0]
                elif sum > 7:
                    img_c[i, j, :] = [0, 0, 0]
                elif sum1 > 7:
                    img_c[i, j, :] = [0, 0, 0]
                else:
                    img_c[i, j, :] = [255, 255, 255]

    img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    _, img_b = cv2.threshold(img_g, 250, 255, cv2.THRESH_BINARY)

    return img_b


def main():
    i = 0
    for build in building:
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
                    r_json = text_detection('./' + build + '/' + file)
                    img_c = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)
                    chambre, cuisine, wc, sdb = data_process(r_json, img_c)
                    scale = df[df['filename'].isin([file])]['m2perpixel']
                    img_seg, a_chambre, a_cuisine, a_wc, a_sdb, wcsdb = room_detection(chambre, cuisine, wc, sdb,
                                                                                       img_b, scale)
                    cv2.imwrite(w_path + file, img_seg)
        i += 1


if __name__ == "__main__":
    main()
