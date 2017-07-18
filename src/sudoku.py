import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from cfg import Cfg


# GLOBALS
cf = Cfg()
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

def on_trackbar_change(x):
    global img,img2,cf
    improve(src=img, dst=img2)
    cf.redraw(img2)


def improve(src, dst):
    global cf
    (a, _) = cf.get_slider('a',on_trackbar_change)
    k = int(a/8*2+3)

    #cv2.bilateralFilter(src,-1,b,c,dst=dst)
    #img3 = img
    #print "blur"
    #cv2.GaussianBlur(src, ksize=(k,k), sigmaX=b, dst=dst)

    (m, _) = cf.get_toggle('m', 1, callback=on_trackbar_change)
    (n, _) = cf.get_toggle('n', 1, callback=on_trackbar_change)

    ddepth = cv2.CV_16S if n == 0 else cv2.CV_8U

    if m == 0:
        print "Kernel:", k-2,
        sys.stdout.flush()
        s16 = cv2.Laplacian(src, ddepth=ddepth, ksize=k-2)

    if m == 1:
        (b, _) = cf.get_slider('dx',on_trackbar_change)
        (c, _) = cf.get_slider('dy',on_trackbar_change)

        if b > 29: b = 29
        if c > 29: c = 29
        if k > 31: k = 31
        if k <= b+1: k = (b+2)/2*2+1
        if k <= c+1: k = (c+2)/2*2+1
        print "Kernel:", k, ", dx:", b+1, "dy:", c+1,
        sys.stdout.flush()
        s16 = cv2.Sobel(src, dx=b+1, dy=c+1, ddepth=ddepth, ksize=k)

    if n == 0:
        as16 = np.absolute(s16)
        dst[:,:] = np.uint8(as16)
    if n == 1:
        dst[:,:] = s16



    # r, _ = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY, dst=img2)
    #cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    #        adaptive_kernel_size, 0, dst=img4)
    print "Done"
    sys.stdout.flush()

def eventloop():
    global cf

    while(1):
        k = cv2.waitKey(1) & 0xFF

        cf.got_key(k)
        if k == 27:
            break

        img = cf.redraw()
        if img is not None:
            #print "Redraw"
            #sys.stdout.flush()
            cv2.imshow('image',img)

    cv2.destroyAllWindows()



init_window()
improve(src=img, dst=img2)
cf.redraw(img2)
eventloop()
