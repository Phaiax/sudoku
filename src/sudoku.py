import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from context import Context


# GLOBALS
cx = Context()
adaptive_kernel_size = 17

def load_image(path, w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h = int( img.shape[1] * w / img.shape[0] )
    return cv2.resize(img, (w, h))



def on_trackbar_change(x):
    global cx
    improve(src=cx.b('orig'))
    cx.redraw()


def calc_scale(shape, long_edge):
    h, w = shape
    scale = float(long_edge) / max(w, h)
    return scale

def resize(src, long_edge):
    h, w = src.shape
    scale = calc_scale(src.shape, long_edge)
    w = int(w * scale)
    h = int(h * scale)
    return (cv2.resize(src, dsize=(w,h)), scale)

def improve(src):
    global cx
    invalid = False
    print "Go",
    sys.stdout.flush()

    # ==================================================================================== RESIZE
    long_edge = 400
    scale = calc_scale(src.shape, long_edge)
    if invalid or cx.once('small'):
        cx['small'], _ = resize(src, long_edge)

    # ==================================================================== Histogram Equalization
    if invalid or cx.once('equi'):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cx['equihist'] = clahe.apply(cx['small'])

    # ==================================================================== Denoise and threshold

    #(h, h_chg) = cx.get_slider('h', on_trackbar_change, 10, 20)
    #(tw, tw_chg) = cx.get_slider('templateWindowSize', on_trackbar_change, 7, 20)
    #(sw, sw_chg) = cx.get_slider('searchWindowSize', on_trackbar_change, 21, 40)
    #if h_chg or tw_chg or sw_chg or invalid or cx.once('denoise'):
        #invalid = True
        #cx['denoise'] = cv2.fastNlMeansDenoising(cx['equihist'], h=h, templateWindowSize=tw, searchWindowSize=sw)
    if invalid or cx.once('denoise'):
        cx['denoise'] = cv2.fastNlMeansDenoising(cx['equihist'], h=16, templateWindowSize=7, searchWindowSize=10)

    #(tr, tr_chg) = cx.get_slider('tr', on_trackbar_change, 10, 40)
    #(cp, cp_chg) = cx.get_slider('C+50', on_trackbar_change, 50, 100)
    #if invalid or tr_chg or cp_chg:
    #    invalid = True
        #cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, tr/2*2+1, C=cp-50)
    if invalid or cx.once('threshold'):
        cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=20)


    # ====================================================================== HOUGH LINE TRANSFORM
    # (a, a_chg) = cx.get_slider('minLineLength', on_trackbar_change, 100)
    # (b, b_chg) = cx.get_slider('maxLineGap', on_trackbar_change, 10)
    # (c, c_chg) = cx.get_slider('threshold', on_trackbar_change, 100)
    # if invalid or a_chg or b_chg or c_chg:
    #     invalid = True
    #     lines = cv2.HoughLinesP(cx['thresholded'],1,np.pi/180,threshold=c,minLineLength=a,maxLineGap=b)
    #     cx['lines'] = np.array(cx['small'])
    #     for x1,y1,x2,y2 in lines[0]:
    #         cv2.line(cx['lines'],(x1,y1),(x2,y2),255,1)


    # ============================================================================= CORNER HARRIS
    # gray = np.float32(cx['thresholded'])
    # (a, a_chg) = cx.get_slider('block_size', on_trackbar_change, 2, 20)
    # (b, b_chg) = cx.get_slider('ksize', on_trackbar_change, 3, 20)
    # (c, c_chg) = cx.get_slider('k', on_trackbar_change, 4, 100)
    # (t, t_chg) = cx.get_slider('t', on_trackbar_change, 4, 100)
    # if a_chg or b_chg or c_chg or t_chg or invalid:
    #     invalid = True
    #     # cx['corners'] = cv2.cornerHarris(gray, blockSize=a, ksize=b/2*2+1, k=float(c)/100)
        # #c = cx['corners']
        # c2 = cv2.resize(cx['corners'], (src.shape[1], src.shape[0]))
        # #c2 = cv2.dilate(c2,None)

        # cx['out'][c2>(float(t)/100)*c2.max()]=255

    # ====================================================================== BETTER CORNER HARRIS
    gray = np.float32(cx['thresholded'])
    # (a, a_chg) = cx.get_slider('minDistance', on_trackbar_change, 10, 250)
    # (b, b_chg) = cx.get_slider('qualityLevel', on_trackbar_change, 1, 100)
    # (c, c_chg) = cx.get_slider('maxCorners', on_trackbar_change, 25, 255)
    # if invalid or a_chg or b_chg or c_chg or t_chg:
    #     invalid = True
    #     corners = cv2.goodFeaturesToTrack(gray, maxCorners=c*10,qualityLevel=float(b)/100,minDistance=a)
    if invalid or cx.once('corners'):
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=2000,qualityLevel=0.25,minDistance=3)
        corners = np.int0(corners)
        cx['out'] = np.array(src)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(cx['out'],(int(x/scale),int(y/scale)),3,255,-1)


    print "Done"
    sys.stdout.flush()
    return


img = load_image(path='assets/test/s1.jpg', w=550)
cx.add_buffer('orig', src=img)
#cx.add_buffer('laplace', shape=img.shape)
#cx.add_buffer('canny', shape=img.shape)
#cx.add_buffer('corners', shape=img.shape)
#cx.add_buffer('out', shape=img.shape)

improve(src=cx.b('orig'))
cx.redraw()
cx.eventloop()
