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


def match_hu(h1, h2, method=cv2.cv.CV_CONTOURS_MATCH_I2):
    m1 = [ np.sign(h) * np.log(h) for h in h1]
    m2 = [ np.sign(h) * np.log(h) for h in h2]
    if method == cv2.cv.CV_CONTOURS_MATCH_I1:
        return sum([ abs( 1/i1 - 1/i2 ) for i1, i2 in zip(m1, m2)])
    if method == cv2.cv.CV_CONTOURS_MATCH_I2:
        return sum([ abs( i1 - i2 ) for i1, i2 in zip(m1, m2)])
    if method == cv2.cv.CV_CONTOURS_MATCH_I3:
        return max([ abs( i1 - i2 ) / abs(i1) for i1, i2 in zip(m1, m2)])

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
        #lines = cv2.HoughLinesP(cx['thresholded'],1,np.pi/180,threshold=c,minLineLength=a,maxLineGap=b)
    if invalid or cx.once('lines'):
        lines = cv2.HoughLinesP(cx['thresholded'],1,np.pi/180,threshold=50,minLineLength=28,maxLineGap=8)
        cx.store('lines', lines[0])
        cx['lines'] = np.array(cx['small'])
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(cx['lines'],(x1,y1),(x2,y2),255,1)


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


    # ============================================================= CONTOURS
    (n, n_chg) = cx.get_slider('biggest n', on_trackbar_change, 10, 40)
    if invalid or cx.once('contours') or n_chg:
        thre_cp = np.array(cx['thresholded'])
        contours, _ = cv2.findContours(thre_cp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        with_area = [(c, cv2.contourArea(c)) for c in contours]
        with_area.sort(key=lambda ca: ca[1])
        contours_filtered = [c[0] for c in with_area[-n:]]



        cx['contours'] = np.array(cx['small'])
        cv2.drawContours(cx['contours'], contours=contours_filtered, contourIdx=-1, color=255, thickness=1)





    # ============================================================= PRECALCULATE POSSIBLE CORNERS
    if invalid or cx.once('precorner'):
        # testing is invariant to rotation, only variant is angle of corner caused by
        # perspective transformation
        precorner = []
        for angle in [45, 65, 75, 90]:
            angle = (angle - 45.) / 180 * np.pi
            im = np.zeros((8,8), np.uint8)
            cv2.line(im,(4,4),(8,4),255,1)
            cv2.line(im,(4,4),(int(4+8*np.sin(angle)),
                               int(4+8*np.cos(angle))),255,1)

            m = cv2.moments(im, binaryImage=True)
            hu = cv2.HuMoments(m)

            precorner.append({'img': im, 'humoments': hu})

            #cx['pre'+str(angle)] = im

        cx.store('precorner', precorner)


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

        # filter corners not on lines
        filtered_corners = []
        lines = cx.load('lines')
        for c in corners:
            for x1,y1,x2,y2 in lines:
                x,y = c.ravel()
                if x1 <= x and x <= x2 and y1 <= y and y <= y2:
                    o = abs((y2-y1)*x-(x2-x1)*y+x2*y1-y2*x1)
                    u = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                    d = float(o)/u
                    if d < 5:
                        filtered_corners.append(c)


        cx['out'] = np.array(src)
        for i in filtered_corners:
            x,y = i.ravel()
            cv2.circle(cx['out'],(int(x/scale),int(y/scale)),3,255,-1)



        # i = 0
        # rows, cols = cx['small'].shape
        # bestcorners = []
        # for corner in filtered_corners:
        #     for blocksize in [4, 8, 16, 32]:
        #         idf=str(blocksize)+str(i)
        #         x,y = corner.ravel()
        #         if (x < blocksize) or (y < blocksize) or (x + blocksize >= cols) or (y+blocksize >= rows):
        #             continue
        #         else:
        #             print "x,y", (x,y),
        #             print "x range", (x-blocksize,x+blocksize),
        #             print "y range", (y-blocksize,y+blocksize),
        #             print "w, h", (cols,rows),
        #             print "ok?", (x < blocksize)
        #             sys.stdout.flush()
        #             sys.stdout.flush()
        #             sys.stdout.flush()
        #             view = cx['thresholded'][y-blocksize:y+blocksize, x-blocksize:x+blocksize]
        #             r = cv2.resize(view, (8, 8))
        #             moments = cv2.moments(r, binaryImage=True)
        #             hu = cv2.HuMoments(moments)
        #             match = max([ match_hu(precorner['humoments'], hu) for precorner in cx.load('precorner')])
        #             bestcorners.append((match, corner, r, idf))
        #     i += 1

        # bestcorners.sort(key=lambda c: c[0])
        # bestcorners = bestcorners[-10:]
        # for m,c,r,idf in bestcorners:
        #     cx['v1'+idf] = r



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
