import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

# GLOBALS
a=10
b=10
c=10
adaptive_kernel_size = 17
redraw = None

def load_image(path, w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h = int( img.shape[1] * w / img.shape[0] )
    img = cv2.resize(img, (w, h))

    return img

img = load_image(path='assets/test/s1.jpg', w=700)

# Buffers
img2 = np.zeros(img.shape, np.uint8)
img3 = np.zeros(img.shape, np.uint8)
img4 = np.zeros(img.shape, np.uint8)


def init_window():
    cv2.namedWindow('image')
    cv2.createTrackbar('a','image',0,255,on_trackbar_change)
    cv2.createTrackbar('b','image',0,255,on_trackbar_change)
    cv2.createTrackbar('c','image',0,255,on_trackbar_change)

def on_trackbar_change(x):
    global a,b,c,img,img2,redraw

    a = cv2.getTrackbarPos('a','image')
    b = cv2.getTrackbarPos('b','image')
    c = cv2.getTrackbarPos('c','image')

    improve(src=img, dst=img2)
    redraw = img2



def improve(src, dst):
    global b,a,c
    k = a/8*2+3
    print "Kernel:", k, ", sigmaColor:", b, "sigmaSpace:", c
    sys.stdout.flush()

    #cv2.bilateralFilter(src,-1,b,c,dst=dst)
    print "Done"
    sys.stdout.flush()
    #img3 = img
    #print "blur"
    #cv2.GaussianBlur(src, ksize=(k,k), sigmaX=b, dst=dst)

    # r, _ = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY, dst=img2)
    #cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    #        adaptive_kernel_size, 0, dst=img4)

def eventloop():
    global redraw

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            pass
        elif k == 27:
            break
        elif redraw is not None:
            print "Redraw"
            sys.stdout.flush()
            cv2.imshow('image',redraw)
            redraw = None

    cv2.destroyAllWindows()



init_window()
improve(src=img, dst=img2)
redraw=img2
eventloop()
