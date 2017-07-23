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
    # (a, a_chg) = cx.get_slider('minLineLength', on_trackbar_change, 28)
    # (b, b_chg) = cx.get_slider('maxLineGap', on_trackbar_change, 8)
    # (c, c_chg) = cx.get_slider('threshold', on_trackbar_change, 50)
    # if invalid or a_chg or b_chg or c_chg:
    #     invalid = True
    #     lines = cv2.HoughLinesP(cx['thresholded'],1,np.pi/180,threshold=c,minLineLength=a,maxLineGap=b)
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
    (n, n_chg) = cx.get_slider('biggest n', on_trackbar_change, 6, 40)
    if invalid or cx.once('contours') or n_chg:
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


        # for c in contours_filtered:
        #     c.shape = (4,2)
        #     x,y = c[0]
        #     cv2.circle(cx['contours'],(int(x),int(y)),4,100,-1)
        #     x,y = c[1]
        #     cv2.circle(cx['contours'],(int(x),int(y)),4,150,-1)
        #     x,y = c[2]
        #     cv2.circle(cx['contours'],(int(x),int(y)),4,200,-1)
        #     x,y = c[3]
        #     cv2.circle(cx['contours'],(int(x),int(y)),4,255,-1)

        #     break


    # ============================================================= CANDIDATES
    if cx.once('math'):
        # for each contour interpret it as the outer square and check how many inner squares are found
        base = np.array([ [1    ,0    ,0    ,0    ],
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
                        ])

        # the `base.dot([pos_0, pos_3, pos_12, pos_15])` results in an array with
        # the points in the square belonging to the following indizes
        # 0  1  2  3
        # 4  5  6  7
        # 8  9  10 11
        # 12 13 14 15
        # squares = [ [0,1,4,5], [1,2,5,6], [2,3,6,7],
        #             [4,5,8,9], [5,6,9,10], [6,7,10,11],
        #             [8,9,12,13], [9,10,13,14], [10,11,14,15] ]

        # # make array to answer the question: If point number `i` is a corner of an inner contour,
        # # which points of `corners` must the other points of the inner contour be?
        # ask = []
        # for i in range(0, 16):
        #     ask_i = []
        #     for square in squares:
        #         if i in square:
        #             remaining_points = list(square)
        #             remaining_points.remove(i)
        #             ask_i.append(remaining_points)
        #     ask.append(ask_i)
        # # Test
        # assert ask[0] == [[1,4,5]]
        # assert ask[1] == [[0,4,5], [2,5,6]]
        # assert ask[5] == [[0,1,4], [1,2,6], [4,8,9], [6,9,10]]

    if invalid or cx.once('find_sudokus'):
        sudokus = []

        for outer in contours_filtered: # N
            outer.shape = (4,2)
            # make all 16 possible corners
            corners = base.dot( outer )
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



            #print "novis", number_of_valid_inner_squares
            if number_of_valid_inner_squares >= 1:
                sudokus.append(outer)
                for x,y in corners:
                    cv2.circle(cx['contours'],(int(x),int(y)),2,0,-1)




        # if a big square has at least one correct inner square, add it to list of squares
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
