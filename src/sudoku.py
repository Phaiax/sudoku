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



def resize(src, long_edge=300):
    h, w = src.shape
    scale = float(long_edge) / max(w, h)
    w = int(w * scale)
    h = int(h * scale)
    return (cv2.resize(src, dsize=(w,h)), scale)

def improve(src):
    global cx
    invalid = False
    print "Go",
    sys.stdout.flush()

    # 1. For sudoku finding, reduce solution to `long_edge` on the long edge
    long_edge = 400
    if cx.once('small'):
        cx['small'], scale = resize(src, long_edge)

    # 2. Histogram Equalization
    if cx.once('equi'):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cx['equihist'] = clahe.apply(cx['small'])

    # 3. Denoise and threshold

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
    if cx.once('threshold')
        cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C=20)


    print "Done"
    sys.stdout.flush()
    return

    (l, l_chg) = cx.get_slider('threshold_lp', on_trackbar_change, 100)
    if l_chg or invalid:
        invalid = True
        cx['laplace'] = cv2.Laplacian(cx['denoise'], ddepth=cv2.CV_8U, ksize=3)
        ret, cx['laplace_edges'] = cv2.threshold(cx['laplace'], l, 255, cv2.THRESH_BINARY_INV)

    if invalid:
        cx['laplace_blured'] = cv2.GaussianBlur(cx['laplace_edges'], ksize=(3, 3), sigmaX=0)


    # 2. Bilateral Filter
    (d, d_chg) = cx.get_slider('sigmaColor', on_trackbar_change, 10, 255)
    (e, e_chg) = cx.get_slider('sigmaSpace/', on_trackbar_change, 40, 100)
    if d_chg or e_chg or invalid:
        invalid = True
        cx['smooth'] = cv2.bilateralFilter(cx['laplace_blured'], d=-1, sigmaColor=d, sigmaSpace=long_edge/e)
        # cx['smooth'] = cx['laplace_edges']



    # #k = int(a/8*2+3)
    # (m, m_chg) = cx.get_toggle('m', 1, callback=on_trackbar_change)
    # (n, n_chg) = cx.get_toggle('n', 1, callback=on_trackbar_change)
    # (_, n_l) = cx.get_toggle('l', 1, callback=on_trackbar_change)



    # edges=cv2.Canny(src, 50, 90, L2gradient=False, edges=cx.b('canny'))



    # (a, a_chg) = cx.get_slider('minLineLength', on_trackbar_change, 100)
    # (b, b_chg) = cx.get_slider('maxLineGap', on_trackbar_change, 10)
    # (c, c_chg) = cx.get_slider('threshold', on_trackbar_change, 100)
    # if a_chg or b_chg or c_chg:
    #     lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=c,minLineLength=a,maxLineGap=b)
    #     cx['lines'][:,:] = cx['orig']
    #     for x1,y1,x2,y2 in lines[0]:
    #         cv2.line(cx['lines'],(x1,y1),(x2,y2),255,1)


    gray = np.float32(cx['smooth'])
    (a, a_chg) = cx.get_slider('block_size', on_trackbar_change, 2, 20)
    (b, b_chg) = cx.get_slider('ksize', on_trackbar_change, 3, 20)
    (c, c_chg) = cx.get_slider('k', on_trackbar_change, 4, 100)
    (t, t_chg) = cx.get_slider('t', on_trackbar_change, 4, 100)
    if a_chg or b_chg or c_chg or t_chg or invalid:
        invalid = True
        # cx['corners'] = cv2.cornerHarris(gray, blockSize=a, ksize=b/2*2+1, k=float(c)/100)
        # #c = cx['corners']
        # c2 = cv2.resize(cx['corners'], (src.shape[1], src.shape[0]))
        # #c2 = cv2.dilate(c2,None)

        # cx['out'][c2>(float(t)/100)*c2.max()]=255

        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners)
        cx['out'] = np.array(src)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(cx['out'],(int(x/scale),int(y/scale)),3,255,-1)


    print "Done"
    sys.stdout.flush()


img = load_image(path='assets/test/s1.jpg', w=550)
cx.add_buffer('orig', src=img)
#cx.add_buffer('laplace', shape=img.shape)
#cx.add_buffer('canny', shape=img.shape)
#cx.add_buffer('corners', shape=img.shape)
#cx.add_buffer('out', shape=img.shape)

improve(src=cx.b('orig'))
cx.redraw()
cx.eventloop()
