import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from context import Context
from itertools import product

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
    invalid = cx.once('invalid')
    print "Go",
    sys.stdout.flush()

    # ==================================================================================== RESIZE
    long_edge = 400
    scale = calc_scale(src.shape, long_edge)
    if invalid:
        cx['small'], _ = resize(src, long_edge)

    # ==================================================================== Histogram Equalization
    if invalid:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cx['equihist'] = clahe.apply(cx['small'])

    # ==================================================================== Denoise and threshold

    #(h, h_chg) = cx.get_slider('h', on_trackbar_change, 10, 20)
    #(tw, tw_chg) = cx.get_slider('templateWindowSize', on_trackbar_change, 7, 20)
    #(sw, sw_chg) = cx.get_slider('searchWindowSize', on_trackbar_change, 21, 40)
    #if h_chg or tw_chg or sw_chg or invalid or cx.once('denoise'):
        #invalid = True
        #cx['denoise'] = cv2.fastNlMeansDenoising(cx['equihist'], h=h, templateWindowSize=tw, searchWindowSize=sw)
    if invalid:
        cx['denoise'] = cv2.fastNlMeansDenoising(cx['equihist'], h=16, templateWindowSize=7, searchWindowSize=10)

    #(tr, tr_chg) = cx.get_slider('tr', on_trackbar_change, 10, 40)
    #(cp, cp_chg) = cx.get_slider('C+50', on_trackbar_change, 50, 100)
    #if invalid or tr_chg or cp_chg:
    #    invalid = True
        #cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, tr/2*2+1, C=cp-50)
    if invalid:
        cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=20)


    # ============================================================= CONTOURS
    (n, n_chg) = cx.get_slider('biggest n', on_trackbar_change, 6, 40)
    if invalid or n_chg:
        invalid = True
        thre_cp = np.array(cx['thresholded'])
        contours, _ = cv2.findContours(thre_cp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        with_area = [(c, cv2.contourArea(c)) for c in contours]
        with_area.sort(key=lambda ca: ca[1])
        contours_filtered = [cv2.convexHull(c[0]) for c in with_area[-n:]]
        epsilon = 5
        contours_filtered = [cv2.approxPolyDP(c,epsilon,closed=True) for c in contours_filtered]
        contours_filtered = np.array([c for c in contours_filtered if len(c) == 4])


        cx['contours'] = np.array(cx['small'])
        cv2.drawContours(cx['contours'], contours=contours_filtered, contourIdx=-1, color=255, thickness=1)

    # ============================================================= CANDIDATES
    if cx.once('math'):
        # for each contour interpret it as the outer square and check how many inner squares are found
        cx.store('base', np.array([ [1    ,0    ,0    ,0    ],
                                  [0.666,0.333,0    ,0    ],
                                  [0.333,0.666,0    ,0    ],
                                  [0    ,1    ,0    ,0    ],
                                  [0.666,0    ,0    ,0.333],
                                  [0.444,0.222,0.111,0.222],
                                  [0.222,0.444,0.222,0.111],
                                  [0    ,0.666,0.333,0    ],
                                  [0.333,0    ,0    ,0.666],
                                  [0.222,0.111,0.222,0.444],
                                  [0.111,0.222,0.444,0.222],
                                  [0    ,0.333,0.666,0    ],
                                  [0    ,0    ,0    ,1    ],
                                  [0    ,0    ,0.333,0.666],
                                  [0    ,0    ,0.666,0.333],
                                  [0    ,0    ,1    ,0    ]
                                ]))

    if invalid:
        sudokus = []

        for outer in contours_filtered: # N
            outer.shape = (4,2)
            # make all 16 possible corners
            corners = cx.load('base').dot( outer )
            # calculate allowed error for inner square points (1/6 of closest corner points)
            outer_next = np.array([outer[1], outer[2], outer[3], outer[0]])
            dists = np.sqrt(np.sum((outer - outer_next)**2, axis=1))
            max_err = 1./6 * np.min(dists)

            number_of_valid_inner_squares = 0

            # compare all contours if they fit to the possible corners (KDTREE?)
            for inner in contours_filtered: # N*N
                inner.shape = (4,2)
                if np.array_equal(inner, outer):
                    continue
                # compare each point of inner with each `corners`
                matches = 0
                for inner_pt, c_pt in product(inner, corners): # N*N*4*16 = 6400 if N = 10
                    dist = np.sum(np.abs(inner_pt - c_pt))
                    if dist < max_err:
                        matches += 1
                        if matches == 4:
                            number_of_valid_inner_squares += 1
                            break

            # if a big square has at least one correct inner square, add it to list of squares
            if number_of_valid_inner_squares >= 1:
                sudokus.append(outer)
                for x,y in corners:
                    cv2.circle(cx['contours'],(int(x),int(y)),2,0,-1)




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
