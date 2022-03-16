import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Vcituvanje na originalnata slika
img = cv.imread('blood.png')
#cv.imshow('1. Originalna slika', img)

# 2. Konverzija vo Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('2. Konverzija vo Grayscale', gray)

# 3. Dodavanje gausov filter na slikata za podobar smoothing
blur = cv.GaussianBlur(gray, (7,7), cv.BORDER_DEFAULT)
#cv.imshow('3. Blured', blur)

# 4. Histogram
hist = cv.equalizeHist(gray)
#cv.imshow('4. Histogram', hist)

# 5. CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
claheNorm = clahe.apply(gray)
#cv.imshow('5. CLAHE', claheNorm)


# 6. Contrast Stretching
def px_val(px, r1, s1, r2, s2):
    if 0 <= px <= r1:
        return (s1 / r1) * px
    elif r1 < px <= r2:
        return ((s2 - s1) / (r2 - r1)) * (px - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (px - r2) + s2


r1 = 100
s1 = 0
r2 = 200
s2 = 255

px_val_vec = np.vectorize(px_val)

contrast_stretch = px_val_vec(gray, r1, s1, r2, s2)
#cv.imshow('6.1. Contrast stretching', contrast_stretch)

contrast_stretch_blur = px_val_vec(blur, r1, s1, r2, s2)
#cv.imshow('6.2. Contrast stretching na slikata so Gausoviot filter', contrast_stretch_blur)

# 7. Detekcija na rabovi
edge = cv.Canny(blur, 10, 152)
#cv.imshow('7. Rabovi na slikata', edge)
edge2 = cv.Canny(blur, 60, 70)
#cv.imshow('7.1. Rabovi CRVENI', edge2)
edge3 = cv.bitwise_xor(edge2,edge)
#cv.imshow('7.2. Rabovi NOVI SAMO CRVENI', edge3)

# 8. Morfoloski operacii na edge
kernel = np.ones((10,10), np.uint8)
kernel1 = np.ones((10,10), np.uint8)
#cv.imshow('8.1. Kernel', kernel)
dialacija = cv.dilate(edge, kernel, iterations=1)
dialacija1 = cv.dilate(edge3, kernel1, iterations=1)
#cv.imshow('8.2. Dialacija', dialacija)
#cv.imshow('8.2.1 Dialacoja na CRVENI', dialacija1)
closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)
closing1 = cv.morphologyEx(edge3, cv.MORPH_CLOSE, kernel1)
#cv.imshow('8.3. Closing', closing)
#cv.imshow('8.3.1 Closing CRVENI', closing1)

# 9. Adaptive Thresholding
th = cv.adaptiveThreshold(edge, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
#cv.imshow('9. Thresholding', th)
th1 = cv.adaptiveThreshold(edge3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
#cv.imshow('9.1. Thresholding CRVENI', th1)

# 10. Inicijalizacija na listiti, iscrtuvanje na krugovi, presmetki
img1 = cv.imread('blood.png', 0)
wbc_golemina, rbc_golemina, wbc_x, wbc_y, rbc_x, rbc_y = [], [], [], [], [], []
display = cv.imread('blood.png')
display1 = cv.imread('blood.png')

krugovi = cv.HoughCircles(edge, cv.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=28, minRadius=1, maxRadius=100)
krugovi1 = cv.HoughCircles(edge3, cv.HOUGH_GRADIENT, 1.58, 20, param1=50, param2=20, minRadius=1, maxRadius=18)
#cv.imshow('10.2. Krugovi', krugovi)

if krugovi is not None:
    krugovi = np.round(krugovi[0,:]).astype(int)

    for x,y,r in krugovi:
        cv.circle(display, (x,y), r, (0,255,0), thickness=2)
        cv.rectangle(display, (x-2, y-2), (x+2, y+2), (0,128,255), -1)
        wbc_golemina.append(r)
        wbc_x.append(x)
        wbc_y.append(y)

    #cv.imshow('10.3. Krugovi na kraj', display)

if krugovi1 is not None:
    krugovi1 = np.round(krugovi1[0, :]).astype(int)

    for x,y,r in krugovi1:
        cv.circle(display1, (x, y), r, (0, 255, 0), thickness=1)
        cv.rectangle(display1, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        rbc_golemina.append(r)
        rbc_x.append(x)
        rbc_y.append(y)

    #cv.imshow('10.3.1 Krugovi na kraj Crveni', display1)

print('\nVkupen broj na beli krvni kletki: ', len(wbc_golemina),'\n')
print('Vkupen broj na crveni krvni kletki: ', len(rbc_golemina),'\n')


# ne bev siguren dali se bara samo radiusot za golemina na kletka ili plostinata, pa gi napisav i dvete
vkupno_beli = 0
for i in wbc_golemina:
    vkupno_beli = vkupno_beli+i
print('Prosecna golemina(radius) na beli krvni kletki: ', vkupno_beli/len(wbc_golemina),' px\n')

vkupno_crveni = 0
for i in rbc_golemina:
    vkupno_crveni = vkupno_crveni + i
print('Prosecna golemina(radius) na crveni krvni kletki: ', vkupno_crveni/len(rbc_golemina),' px\n')

prosecna_plostina_beli = (vkupno_beli/len(wbc_golemina))**2 * 3.14
prosecna_plostina_crveni = (vkupno_crveni/len(rbc_golemina))**2 * 3.14

print('Prosecna golemina(plostina) na beli krvni kletki: ',prosecna_plostina_beli,' px\n')
print('Prosecna golemina(plostina) na crveni krvni kletki: ', prosecna_plostina_crveni,' px\n')


print('Soodnos beli/crveni krvni kletki: ', len(wbc_golemina)/len(rbc_golemina),'\n')

lokacii_beli = [(wbc_x[i], wbc_y[i]) for i in range(0,len(wbc_golemina))]
print('Lista na lokacii na belite krvni kletki: ', lokacii_beli,'\n')

lokacii_crveni = [(rbc_x[i], rbc_y[i]) for i in range(0,len(rbc_golemina))]
print('Lista na lokacii na crvenite krvni kletki: ', lokacii_crveni,'\n')

cv.imshow('Beli krvni kletki na slikata', display)

cv.waitKey(0)
