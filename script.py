import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('dave.jpg', 0)
# img = cv2.medianBlur(img, 5)

for i in range(4):
    img = cv2.imread('recipts/reciptTest' + str(i + 1) + '.JPG', 0)
    img = cv2.medianBlur(img, 5)

    ret, th1 = cv2.threshold(img, 95, 195, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    titles = [
        'Original Image', 'Global Thresholding (v = 95)',
        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding'
    ]
    images = [img, th1, th2, th3]

    for x in range(4):
        plt.subplot(2, 2, x + 1), plt.imshow(images[x], 'gray')
        plt.title(titles[x])
        plt.xticks([]), plt.yticks([])
    plt.show()

    # img = cv.imread('recipts/reciptTest' + str(i + 1) + '.JPG', 0)
    # blur = cv.GaussianBlur(img, (5, 5), 0)
    # # find normalized_histogram, and its cumulative distribution function
    # hist = cv.calcHist([blur], [0], None, [256], [0, 256])
    # hist_norm = hist.ravel() / hist.max()
    # Q = hist_norm.cumsum()
    # bins = np.arange(256)
    # fn_min = np.inf
    # thresh = -1
    # for i in range(1, 256):
    #     p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
    #     q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
    #     b1, b2 = np.hsplit(bins, [i])  # weights
    #     # finding means and variances
    #     m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
    #     v1, v2 = np.sum(((b1 - m1)**2) * p1) / q1, np.sum(
    #         ((b2 - m2)**2) * p2) / q2
    #     # calculates the minimization function
    #     fn = v1 * q1 + v2 * q2
    #     if fn < fn_min:
    #         fn_min = fn
    #         thresh = i
    # # find otsu's threshold value with OpenCV function
    # ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # print("{} {}".format(thresh, ret))

    # img = cv.imread('noisy2.png',0)
    # # global thresholding
    # ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # # Otsu's thresholding
    # ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv.GaussianBlur(img,(5,5),0)
    # ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # # plot all the images and their histograms
    # images = [img, 0, th1,
    #         img, 0, th2,
    #         blur, 0, th3]
    # titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
    #         'Original Noisy Image','Histogram',"Otsu's Thresholding",
    #         'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    # for i in xrange(3):
    #     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    # plt.show()
