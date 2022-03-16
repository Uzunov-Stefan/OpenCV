import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# 1. Template matching за наоѓање на објект на дадените слики
template = cv.imread('template.png', 0)
im1 = cv.imread('lena.png',0)

plt.plot()
# plt.imshow(template, cmap='gray')
# plt.show()
#
# plt.imshow(im1, cmap='gray')
# plt.show()


methods = [cv.TM_SQDIFF, cv.TM_CCOEFF, cv.TM_CCORR]

coord_tl = []
coord_br=[]

for method in methods:
    res = cv.matchTemplate(im1, template, method)
    # plt.imshow(res, cmap='gray')
    # plt.show()
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
    #print(min_loc,max_loc)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        topleft = min_loc
    else:
        topleft = max_loc

    bottomright = (topleft[0]+50, topleft[1]+50)

    #print(topleft)
    #print(bottomright)

    coord_tl.append(topleft)
    coord_br.append(bottomright)

#print(coord_tl)
#print(coord_br)

for i in range(3):
    cv.rectangle(im1, coord_tl.pop(), coord_br.pop(), 255, 4)
    # plt.plot()
    # plt.imshow(im1, cmap='gray')
    # plt.show()

plt.imshow(im1, cmap='gray')
plt.show()
cv.imshow('1. Template matching', im1)


# 2. од знамето да се најдат сите линии и да се нацртаат со зелена боја
im2 = cv.imread('makedonija.png', 1)
im22 = cv.cvtColor(im2, cv.COLOR_BGR2RGB)

edge = cv.Canny(im22, 50, 100)

# plt.imshow(edge, cmap='gray')
# plt.show()

output1 = im2.copy()

lines = cv.HoughLinesP(edge,1, np.pi/180, 30, minLineLength=15, maxLineGap=20)


for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(output1,(x1,y1),(x2,y2),(0,255,0),2)

plt.imshow(output1)
plt.show()

cv.imshow('2. Site linii vo zelena boja', output1)


# 3. да се најде кругот и да се нацрта со зелена боја
output2 = im2.copy()
gray = cv.cvtColor(output2, cv.COLOR_BGR2GRAY)

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,1,20,param1=50,param2=50,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for c in circles[0,:]:
    cv.circle(output2,(c[0],c[1]),c[2],(0,255,0),2)
    cv.circle(output2, (c[0], c[1]), 2, (255, 255, 255), 2)

plt.imshow(output2)
plt.show()

cv.imshow('3. Krugot so zelena boja', output2)


# 4.1. сегментација на објектите на сликата од знамето со користење на watershed
output3 = im2.copy()
gray1 = cv.cvtColor(output3, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# plt.imshow(thresh, cmap='gray')
# plt.show()

kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# plt.imshow(opening, cmap='gray')
# plt.show()

sure_bg = cv.dilate(thresh, kernel, iterations=5)
# plt.imshow(sure_bg, cmap='gray')
# plt.show()

dist_tr = cv.distanceTransform(opening, cv.DIST_L2, 5)

ret2, sure_fg = cv.threshold(dist_tr, 0.5*dist_tr.max(), 255, 0)
# plt.imshow(sure_fg, cmap='gray')
# plt.show()

sure_fg = np.uint8(sure_fg)
sure_bg = np.uint8(sure_bg)

unknown = cv.subtract(sure_bg, sure_fg)
# plt.imshow(unknown, cmap='gray')
# plt.show()

ret3, markers = cv.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
# plt.imshow(markers)
# plt.show()

markers = cv.watershed(output3, markers)
output3[markers==-1] = [0,255,0]
plt.imshow(output3)
plt.show()
plt.imshow(markers)
plt.show()
cv.imshow('4.1. segmentacija so watershed na pozadina', output3)

# 4.2. сегментација на објектите на сликата од знамето со користење на watershed
output4 = im2.copy()
gray2 = cv.cvtColor(output4, cv.COLOR_BGR2GRAY)
output4 = cv.cvtColor(output4, cv.COLOR_BGR2RGB)

# lower_color_bounds = cv.Scalar(100, 100, 0)
# upper_color_bounds = cv.Scalar(255 , 255,100)

mask = cv.inRange(output4, (100, 100, 0), (255 , 255,100))
mask_rgb = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)

extracted = output4 & mask_rgb
# cv.imshow('test', extracted)
extracted = cv.cvtColor(extracted, cv.COLOR_RGB2BGR)
# cv.imshow('test', extracted)
gray2 = cv.cvtColor(extracted, cv.COLOR_BGR2GRAY)
# cv.imshow('gray2', gray2)

ret4, thresh1 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# plt.imshow(thresh1)
# plt.show()

kernel1 = np.ones((3,3), np.uint8)
opening1 = cv.morphologyEx(thresh1, cv.MORPH_OPEN, kernel1, iterations=2)
# plt.imshow(opening1, cmap='gray')
# plt.show()

sure_bg1 = cv.dilate(thresh1, kernel1, iterations=5)
# plt.imshow(sure_bg1, cmap='gray')
# plt.show()

dist_tr1 = cv.distanceTransform(opening1, cv.DIST_L2, 5)

ret5, sure_fg1 = cv.threshold(dist_tr1, 0.1*dist_tr1.max(), 255, 0)
# plt.imshow(sure_fg1, cmap='gray')
# plt.show()

sure_fg1 = np.uint8(sure_fg1)
sure_bg1 = np.uint8(sure_bg1)

unknown1 = cv.subtract(sure_bg1, sure_fg1)
# plt.imshow(unknown1, cmap='gray')
# plt.show()

ret6, markers1 = cv.connectedComponents(sure_fg1)
markers1 = markers1+1
markers1[unknown1==255] = 0
# plt.imshow(markers1)
# plt.show()

# output4 = cv.cvtColor(output4, cv.COLOR_BGR2RGB)

markers1 = cv.watershed(output4, markers1)
output4[markers1==-1] = [0,255,0]
plt.imshow(output4)
plt.show()
plt.imshow(markers1)
plt.show()

output4 = cv.cvtColor(output4, cv.COLOR_BGR2RGB)

cv.imshow('4.2. segmentacija so watershed na kraci i sonce', output4)

cv.waitKey(0)