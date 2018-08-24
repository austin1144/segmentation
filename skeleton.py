''' Cell counting '''


import cv2
import numpy as np


def imshow(name, images):
    ''' Display images (a list with images all equal number of channels) all together '''
    image = np.concatenate(images, axis=1)
    image = cv2.resize(image, dsize=tuple([s // 2 for s in image.shape if s > 3])[::-1])
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    
def detect_edges(img):
    ''' Canny edge detection '''  
    ########
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_binary = cv2.Canny(img, 100, 150)
    edges_on_grayscale_img =  cv2.Canny(grayscale_img , 100, 125)
    ########
    imshow('Processed Images', [grayscale_img, edges_binary, edges_on_grayscale_img])

    
def detect_circles(img):
    ''' Hough transform to detect circles ''' 
    ########
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.medianBlur(grayscale_img, 7)
    rows = img2.shape[0]
    hough_transform = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 1, (rows/12), param1= 100, param2 = 10, minRadius=3, maxRadius=80)
                                                    #  inver. ratio of resolution, (rows/12) minmum distance, param1= for edge   parameter2 threshold for center circle
    circles_on_original_img = img
    if hough_transform is not None:
        hough_transform = np.uint16(np.around(hough_transform))
        for i in hough_transform[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # print center
            # circle center
            cv2.circle(circles_on_original_img, center, 1, (0, 100, 100), 3)
            # circle outline
            cv2.circle(circles_on_original_img, center, radius, (255, 0, 0), 3)
            # print radius  #5-30
    ########
    imshow('All Detected Circles', [circles_on_original_img])
    return hough_transform
        
    
def calculate_features(img, circles):
    ''' Use the Hough transform to derive a feature vector for each circle '''
    ########
    ct = 150
    datos_r = np.zeros(ct)
    datos_g = np.zeros(ct)
    datos_b = np.zeros(ct)

    mask_b = np.zeros(img.shape, np.uint8)
    j = 0
    for i in circles[0, :]:
        mask = np.zeros(img.shape, np.uint8)
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        img_features = img*mask
        datos_b[j],datos_g[j],datos_r[j] = (np.sum(img_features[:,:,0])/np.count_nonzero(img_features[:,:,0])),(np.sum(img_features[:,:,1])/np.count_nonzero(img_features[:,:,1])),(np.sum(img_features[:,:,0])/np.count_nonzero(img_features[:,:,2]))
        j += 1
    ########
    print j
    return img_features

def threshold_circles(img, circles, features, thresholds):
    ''' Threshold the feature vector to get the "right" circles '''
    ########
    ##TODO##
    # print "here"
    # imshow('second', [img])
    # ORANGE_MIN = np.array([10, 10, 10], np.uint8)  #
    # ORANGE_MAX = np.array([100, 140, 106], np.uint8)
    b = range(thresholds[0][0],thresholds[0][1])
    g = range(thresholds[1][0], thresholds[1][1])
    r = range(thresholds[2][0], thresholds[2][1])
    n = 0
    for i in circles[0, :]:
        mask = np.zeros(img.shape, np.uint8)
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        img_features = img * mask
        datos_b, datos_g, datos_r = (
                                             np.sum(img_features[:, :, 0]) / np.count_nonzero(img_features[:, :, 0])), (
                                             np.sum(img_features[:, :, 1]) / np.count_nonzero(img_features[:, :, 1])), (
                                             np.sum(img_features[:, :, 0]) / np.count_nonzero(img_features[:, :, 2]))
        # print img_features
        if datos_b in b and datos_g in g and datos_r in r:
            n += 1
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            cv2.circle(img, center, radius, (0, 0, 255), 3)
    selected_circles_on_original_image = img

    # selected_circles_on_original_image = cv2.inRange(img, ORANGE_MIN, ORANGE_MAX)
    ########
    imshow('Only Selected Circles', [selected_circles_on_original_image])
    return n
       
 
if __name__ == '__main__':
    
    #read the image
    img = cv2.imread('normal.jpg')
    #show the image
    imshow('Original Image', [img])
    #do detection
    # detect_edges(img)
    circles = detect_circles(img)
    features = calculate_features(img, circles)
    img = cv2.imread('normal.jpg')
    n = threshold_circles(img, circles, features, ((10, 100), (10, 140), (10, 106)))

    #print result
    print("We counted {} cells.".format(n))