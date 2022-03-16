import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# # read image and display it
#
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# cv.waitKey(0)



# # read video and display it
# # capture images from videocamera. Capture and display each frame in while loop.
# # make a break with if (when 'd' is pressed)
#
# capture = cv.VideoCapture(1)
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)
#
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
#
# capture.release()
# cv.destroyAllWindows()



# # resizing and rescaling
#
# #resizing function
# def rescaleFrame(frame, scale=0.2):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#
#     dimensions = (width, height)
#
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
#
# # image resizing
# img = cv.imread('istock.jpg')
#
# img_resized = rescaleFrame(img)
#
# cv.imshow('Doggo', img)
# cv.imshow('Doggo resized', img_resized)
#
# cv.waitKey(0)
#
# # video resizing
# capture = cv.VideoCapture(1)
# while True:
#     isTrue, frame = capture.read()
#
#     frame_resized = rescaleFrame(frame)
#
#     cv.imshow('Video', frame)
#     cv.imshow('Video Resized', frame_resized)
#
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
#
# capture.release()
# cv.destroyAllWindows()
#
# # specific resizing for video(live video)
# # doesn't work the way it's supposed to
# def changeRes(width, height):
#     capture.set(3,width)
#     capture.set(4,height)



# # Drawing shapes and putting text
# #
# blank = np.zeros((500,500,3),dtype='uint8')
# cv.imshow('Blank', blank)
#
# # 1. Paint the image a certain color
# blank[200:300, 300:400] = 0,255,0
# cv.imshow('Green', blank)
#
# # 2. Draw a rectangle
# cv.rectangle(blank, (0,0), (blank.shape[1]//3, blank.shape[0]//2), (255,0,0), thickness=-1)
# cv.imshow('Rectangle',blank)
#
# # 3. Draw a circle
# cv.circle(blank, (250,250), 40, (0,0,255), thickness=3)
# cv.imshow('Circle', blank)
#
# # 4. Draw a line
# cv.line(blank, (0,0), (270,270), (255,255,0), thickness=5)
# cv.imshow('Line',blank)
#
# # 5. Write text
# cv.putText(blank, 'Hello', (300,400), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
# cv.imshow('Text', blank)
#
# cv.waitKey(0)



# # Essential functions in OpenCV
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# # Convert to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # Blur
# blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)
#
# # Edge Cascade
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)
#
# # Dilating the image
# dialated = cv.dilate(canny, (7,7), iterations=3)
# cv.imshow('Dialated', dialated)
#
# # Eroding
# eroded = cv.erode(dialated, (7,7), iterations=3)
# cv.imshow('Eroded', eroded)
#
# # Resize
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)
#
# # Cropping
# cropped = img[50:200, 200:400]
# cv.imshow('Cropped', cropped)
#
# cv.waitKey(0)



# # Image Transformations
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# # Translation
# def translate(img, x, y):
#     transMat = np.float32([[1,0,x],[0,1,y]])
#     dimensions = (img.shape[1], img.shape[0])
#
#     return cv.warpAffine(img, transMat, dimensions)
#
# translated = translate(img, 100, 100)
# cv.imshow('Translated(x,y)', translated)
#
# # Rotation
# def rotate(img, angle, rotPoint = None):
#     (height, width) = img.shape[:2]
#
#     if rotPoint is None:
#         rotPoint = (width//2,height//2)
#
#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
#     dimensions = (width,height)
#
#     return cv.warpAffine(img, rotMat, dimensions, )
#
# rotated = rotate(img, -45)
# cv.imshow('Rotated', rotated)
#
# # Resizing
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)
#
# # Flip
# flip0 = cv.flip(img, 0)
# cv.imshow('Flip 0', flip0)
# flip1 = cv.flip(img, 1)
# cv.imshow('Flip 1', flip1)
# flipNeg1 = cv.flip(img, -1)
# cv.imshow('Flip -1', flipNeg1)
#
# # Cropping
# cropped = img[200:300, 300:400]
# cv.imshow('Cropped', cropped)
#
# cv.waitKey(0)



# # Contour Detection
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow('Blank', blank)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # # method 1
# # blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# # cv.imshow('Blur', blur)
# # canny = cv.Canny(blur, 125, 175)
# # cv.imshow('Canny', canny)
#
# # method 2
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)
#
# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# print(f'{len(contours)} contour(s) found!')
#
# cv.drawContours(blank, contours, -1, (0,0,255), thickness=1)
# cv.imshow('Drawing contours', blank)
#
# cv.waitKey(0)



# # Color Spaces
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# # BGR to Grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
# cv.imshow('HSV', hsv)
#
# # BGR to LAB
# lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
# cv.imshow('LAB', lab)
#
# # plot shows image as RGB, openCV reads images as BGR by default
# plt.imshow(img)
# plt.show()
#
# # BGR to RGB for plot
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.imshow(rgb)
# plt.show()
#
# # HSV to BGR
# hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow('HSV 2 BGR', hsv_bgr)
# # can convert bgr to anything and anything to bgr, cant do hsv to lab directly, has to go through bgr
#
# cv.waitKey(0)



# # Color Channels
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# # Split image into channels
# b,g,r = cv.split(img)
#
# cv.imshow('Blue', b)
# cv.imshow('Green', g)
# cv.imshow('Red', r)
#
# print(img.shape)
# print(b.shape)
# print(g.shape)
# print(r.shape)
#
# # merge color channels
# merged = cv.merge([b,g,r])
# cv.imshow('Merged', merged)
#
# # Show actual colors in different channels
# blank = np.zeros(img.shape[:2], dtype='uint8')
# blue = cv.merge([b,blank,blank])
# green = cv.merge([blank,g,blank])
# red = cv.merge([blank,blank,r])
#
# cv.imshow('Blue Actual', blue)
# cv.imshow('Green Actual', green)
# cv.imshow('Red Actual', red)
#
# cv.waitKey(0)



# # Blurring Techniques
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# # Averaging
# average = cv.blur(img, (3,3))
# cv.imshow('Average Blur', average)
#
# # Gaussian Blur
# gauss = cv.GaussianBlur(img, (3,3), 0)
# cv.imshow('Gaussian Blur', gauss)
#
# # Median Blur
# median = cv.medianBlur(img, 3)
# cv.imshow('Median Blur', median)
#
# # Bilateral Blur
# bilateral = cv.bilateralFilter(img, 10, 35, 25)
# cv.imshow('Bilateral', bilateral)
#
# cv.waitKey(0)



# # BITWISE Operators
# #
# blank = np.zeros((400,400), dtype='uint8')
#
# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)
#
# cv.imshow('Rectangle', rectangle)
# cv.imshow('Circle', circle)
#
# # BITWISE AND
# bitwise_and = cv.bitwise_and(rectangle, circle)
# cv.imshow('Bitwise AND', bitwise_and)
#
# # BITWISE OR
# bitwise_or = cv.bitwise_or(rectangle, circle)
# cv.imshow('Bitwise OR', bitwise_or)
#
# # BITWISE XOR
# bitwise_xor = cv.bitwise_xor(rectangle, circle)
# cv.imshow('Bitwise XOR', bitwise_xor)
#
# # BITWISE NOT
# bitwise_not = cv.bitwise_not(circle)
# cv.imshow('Bitwise NOT', bitwise_not)
#
# cv.waitKey(0)



# # Masking
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank', blank)
#
# mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# cv.imshow('Mask', mask)
#
# masked = cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Masked Image', masked)
#
# cv.waitKey(0)



# # Computing Histograms
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# blank = np.zeros(img.shape[:2],dtype='uint8')
# cv.imshow('Blank', blank)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
# cv.imshow('Mask', mask)
#
# masked = cv.bitwise_and(gray, gray, mask=mask)
# cv.imshow('Masked', masked)
#
# # Grayscale Histogram
# gray_hist = cv.calcHist([gray],[0],mask, [256], [0,256])
#
# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of px')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()
#
# # Color Histogram
# mask1 = cv.bitwise_and(img,img, mask=mask)
# cv.imshow('Mask1', mask1)
#
# plt.figure()
# plt.title('Color Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of px')
# colors = ('b', 'g', 'r')
# for i,col in enumerate(colors):
#     hist = cv.calcHist([img], [i], mask, [256], [0,256])
#     plt.plot(hist, color=col)
#     plt.xlim([0,256])
#
# plt.show()
#
# cv.waitKey(0)



# # Thresholding
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # Simple Thresholding
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# cv.imshow('Simple Thresholding', thresh)
#
# threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simple Thresholding Inverse', thresh_inv)
#
# # Adaptive Thresholding
# adaptive_thresh_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 9)
# cv.imshow('Adaptive Thresholding Mean', adaptive_thresh_mean)
#
# adaptive_thresh_inv_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 9)
# cv.imshow('Adaptive Thresholding Inverse Mean', adaptive_thresh_inv_mean)
#
# adaptive_thresh_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 9)
# cv.imshow('Adaptive Thresholding Gaussian', adaptive_thresh_gaussian)
#
# adaptive_thresh_inv_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 9)
# cv.imshow('Adaptive Thresholding Inverse Gaussian', adaptive_thresh_inv_gaussian)
#
# cv.waitKey(0)



# # Edge Detection
# #
# img = cv.imread('istock.jpg')
# cv.imshow('Doggo', img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # Laplacian
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)
#
# # Sobel
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
#
# cv.imshow('Sobel X', sobelx)
# cv.imshow('Sobel Y', sobely)
#
# sobel_combined = cv.bitwise_or(sobelx, sobely)
#
# cv.imshow('Sobel Combined XY', sobel_combined)
#
# # Canny
# canny = cv.Canny(gray, 150, 175)
# cv.imshow('Canny', canny)
#
# cv.waitKey(0)



# Face Detection
#
