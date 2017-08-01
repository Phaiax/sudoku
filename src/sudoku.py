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


w_sudoku = 150
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
font_thickness = 1

def improve(src):
    global cx
    invalid = cx.once('invalid')
    print "Go",
    sys.stdout.flush()

    # ==================================================================================== Resize

    long_edge = 400
    scale = calc_scale(src.shape, long_edge)
    if invalid:
        cx['small'], _ = resize(src, long_edge)

    # ==================================================================== Histogram Equalization

    if invalid:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cx['equihist'] = clahe.apply(cx['small'])

    # ==================================================================== Denoise and Threshold

    #(h, h_chg) = cx.get_slider('h', on_trackbar_change, 10, 20)
    #(tw, tw_chg) = cx.get_slider('templateWindowSize', on_trackbar_change, 7, 20)
    #(sw, sw_chg) = cx.get_slider('searchWindowSize', on_trackbar_change, 21, 40)
    #if h_chg or tw_chg or sw_chg or invalid or cx.once('denoise'):
        #invalid = True
        #cx['denoise'] = cv2.fastNlMeansDenoising(cx['equihist'], h=h, templateWindowSize=tw, searchWindowSize=sw)
    if invalid:
        cx['denoise'] = cv2.fastNlMeansDenoising(cx['equihist'],
                                                 h=16,
                                                 templateWindowSize=7,
                                                 searchWindowSize=10)

    #(tr, tr_chg) = cx.get_slider('tr', on_trackbar_change, 10, 40)
    #(cp, cp_chg) = cx.get_slider('C+50', on_trackbar_change, 50, 100)
    #if invalid or tr_chg or cp_chg:
    #    invalid = True
    #    cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'],
    #     255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, tr/2*2+1, C=cp-50)
    if invalid:
        cx['thresholded'] = cv2.adaptiveThreshold(cx['denoise'],
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, C=20)


    # ===================================================================== Global Square Contours

    (n, n_chg) = cx.get_slider('biggest n', on_trackbar_change, 6, 40)
    if invalid or n_chg:
        invalid = True
        thre_cp = np.array(cx['thresholded'])
        contours, _ = cv2.findContours(thre_cp,
            mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        with_area = [(c, cv2.contourArea(c)) for c in contours]
        with_area.sort(key=lambda ca: ca[1])
        contours_filtered = [cv2.convexHull(c[0]) for c in with_area[-n:]]
        epsilon = 5
        contours_filtered = [cv2.approxPolyDP(c,epsilon,closed=True)
                             for c in contours_filtered]
        contours_filtered = np.array([c for c in contours_filtered
                                      if len(c) == 4])


        cx['contours'] = np.array(cx['small'])
        cv2.drawContours(cx['contours'], contours=contours_filtered,
                         contourIdx=-1, color=255, thickness=1)

    # ================================================== Base Matrix for Calculation Of Corners

    if cx.once('math'):
        cx.store('base', np.array([
            [1    ,0    ,0    ,0    ],
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

    # ====================================================== Validate Squares -> Sudoku Candidate

    if invalid:
        sudokus = []

        # for each contour interpret it as the outer
        # square and check how many inner squares are found

        for outer in contours_filtered: # N
            outer.shape = (4,2)
            # make all 16 possible corners
            corners = cx.load('base').dot( outer )
            # calculate allowed error for inner square points
            # (1/6 of closest corner points)
            outer_next = np.array([outer[1], outer[2], outer[3], outer[0]])
            dists = np.sqrt(np.sum((outer - outer_next)**2, axis=1))
            max_err = 1./6 * np.min(dists)

            number_of_valid_inner_squares = 0

            # compare all contours if they fit to
            # the possible corners (KDTREE?)
            for inner in contours_filtered: # N*N
                inner.shape = (4,2)
                if np.array_equal(inner, outer):
                    continue
                # compare each point of inner with each `corners`
                matches = 0
                # N*N*4*16 = 6400 if N = 10
                for inner_pt, c_pt in product(inner, corners):
                    dist = np.sum(np.abs(inner_pt - c_pt))
                    if dist < max_err:
                        matches += 1
                        if matches == 4:
                            number_of_valid_inner_squares += 1
                            break

            if number_of_valid_inner_squares >= 1:
                sudokus.append(outer)
                for x,y in corners:
                    cv2.circle(cx['contours'],(int(x),int(y)),2,0,-1)

    # ==================================================== Copy/Warp Sudoku from Original Picture

    if invalid and len(sudokus) > 0:
        pts1 = np.float32(sudokus[0]* (1./scale))
        w = w_sudoku
        pts2 = np.float32([[0,0],[w,0],[w,w],[0,w]])
        pts2 = np.float32([[w,w],[0,w],[0,0],[w,0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        cx['s1_raw'] = cv2.warpPerspective(src, M,
            dsize=(w,w), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # ================================================================== Equalization and Denoise

    #(cl, cl_chg) = cx.get_slider('clipLimit', on_trackbar_change, 20, 100)
    #(tgs, tgs_chg) = cx.get_slider('tileGridSize', on_trackbar_change, 8, 30)
    #if invalid or cl_chg or tgs_chg:
    #    invalid = True
    #    if tgs == 0: tgs = 1
    #    clahe = cv2.createCLAHE(clipLimit=float(cl)/10, tileGridSize=(tgs,tgs))
    if invalid:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cx['s1_equihist'] = clahe.apply(cx['s1_raw'])
        sudoku = cv2.fastNlMeansDenoising(cx['s1_equihist'], h=16,
            templateWindowSize=7, searchWindowSize=10)
        sudoku = clahe.apply(sudoku)

        cx['s1'] = sudoku


    # ============================================= Remove Borders and Noise, Get Number Contours

    d = w_sudoku / 9
    def contour_meta(n):
        x,y,w,h = cv2.boundingRect(n)
        area = w*h
        aspect_ratio = float(w)/h
        center_x = x + float(w)/2
        center_y = y + float(h)/2
        c = int(np.round(center_x / d)) - 1 # 0 based
        r = int(np.round(center_y / d)) - 1 # 0 based
        return {
                'r': r, # one based
                'c': c,
                'x': x,
                'y': y,
                'center_x': center_x,
                'center_y': center_y,
                'w': w,
                'h': h,
                'contour': n,
                'inners': [],
                'num': 0,
                'area': area,
                'aspect_ratio': aspect_ratio
                }

    if invalid:
        cx['s1_thresholded'] = cv2.adaptiveThreshold(cx['s1'], 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=20)
        thre_cp = np.array(cx['s1_thresholded'])
        contours, _ = cv2.findContours(thre_cp, mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)

        cx['s1_numbers_unfiltered'] = np.zeros(cx['s1'].shape, np.uint8)
        cv2.drawContours(cx['s1_numbers_unfiltered'], contours=contours, contourIdx=-1, color=255, thickness=1)


        outer_contours = []
        mini_contours = []
        number_contours = []
        area_threshold_max = (w_sudoku / 9)**2
        area_threshold_min = (w_sudoku / 9)**2 / 50
        for c in contours:
            c = contour_meta(c)

            if c['aspect_ratio'] > 3 \
                or c['area'] > area_threshold_max \
                or c['area'] < area_threshold_min:
                outer_contours.append(c)
            elif c['aspect_ratio'] < 1 and c['aspect_ratio'] > 0.3:
                number_contours.append(c)
            elif c['aspect_ratio'] < 1.5 and c['aspect_ratio'] > 0.5:
                mini_contours.append(c)



        cx.store('number_contours', number_contours)
        cx.store('mini_contours', mini_contours)

        cx['s1_cleared'] = np.array(cx['s1_thresholded'])

        cv2.fillPoly(cx['s1_cleared'], pts=[c['contour'] for c in outer_contours], color=0)
        cv2.line(cx['s1_cleared'], pt1=(0,0), pt2=(0,w_sudoku),
                 color=0, thickness=2)
        cv2.line(cx['s1_cleared'], pt1=(0,0), pt2=(w_sudoku,0),
                 color=0, thickness=2)
        cv2.line(cx['s1_cleared'], pt1=(w_sudoku,0), pt2=(w_sudoku,w_sudoku),
                 color=0, thickness=2)
        cv2.line(cx['s1_cleared'], pt1=(0,w_sudoku), pt2=(w_sudoku,w_sudoku),
                 color=0, thickness=2)

        cx['s1_numbers0'] = np.zeros(cx['s1'].shape, np.uint8)
        cv2.drawContours(cx['s1_numbers0'], contours=[ c['contour'] for c in mini_contours ], contourIdx=-1, color=255, thickness=1)


    # ====================================================================== Advanced Deformation
    # find deformation information
    # maybe implement later, for now it is good enough


    # ========================================================= Rotation!


    # ========================================================= Position and Size for Each Number

    if invalid:
        stat = {'max_h': 1, 'min_h': d, 'sum_h': 0}
        numbers = {}
        for n in cx.load('number_contours'):

            cv2.circle(cx['s1'],(int(d*(n['c']+1)-d/3),int(d*(n['r']+1)-d/3)),d/3,0,-1)
            cv2.circle(cx['s1'],(int(n['center_x']),int(n['center_y'])),d/5,50,-1)

            r,c = (n['r'], n['c'])
            # remove duplicates, take largest
            if (r,c) in numbers:
                #print "duplicate at ", (r,c)
                prev = numbers[(r,c)]
                if prev['h'] * prev['w'] > n['w'] * n['h']:
                    prev['inners'].append(n)
                    continue
                else:
                    n.append(prev)
                    stat['sum_h'] -= prev['h']

            numbers[(r,c)] = n
            stat['max_h'] = max(n['h'], stat['max_h'])
            stat['min_h'] = min(n['h'], stat['min_h'])
            stat['sum_h'] += n['h']

        stat['avg_h'] = stat['sum_h'] / len(numbers)
        cx.store('font_contour_h', stat['avg_h'])
        cx.store('numbers', numbers)
        print stat

    # ========================================================= Merge with mini contours
    if invalid:
        numbers = cx.load('numbers')
        for mini in cx.load('mini_contours'):
            if (mini['r'], mini['c']) in numbers:
                number = numbers[(mini['r'], mini['c'])]
                # print "same same", (number['x'], "<=", mini['x']), (number['y'], "<=", mini['y']), \
                #         (number['x'] + number['w'], '>=', mini['x'] + mini['w']), \
                #         (number['y'] + number['h'], '>=', mini['y'] + mini['h'])
                if number['x'] <= mini['x'] \
                    and number['y'] <= mini['y'] \
                    and number['x'] + number['w'] >= mini['x'] + mini['w'] \
                    and number['y'] + number['h'] >= mini['y'] + mini['h']:
                    number['inners'].append(mini)
                    # print "app"

        cx['s1_numbers'] = np.zeros(cx['s1'].shape, np.uint8)
        for _, c in numbers.items():
            cv2.drawContours(cx['s1_numbers'], contours=[c['contour']], contourIdx=-1, color=255, thickness=1)
            for i in c['inners']:
                cv2.drawContours(cx['s1_numbers'], contours=[i['contour']], contourIdx=-1, color=255, thickness=1)


    # ==================================================================  Get numbers
    # 'contour': array([[[54, 24]], [[54, 27]], [[55, 28]], [[55, 29]],
    #                  [[54, 30]], [[54, 32]], [[55, 32]], [[56, 33]], [[58, 33]], [[59, 32]],
    #                     [[59, 29]], [[58, 28]], [[59, 27]], [[59, 25]], [[58, 24]]], dtype=int32),
    # 'center_y': 29.0,
    # 'num': 0,
    # 'c': 3,
    # 'h': 10,
    # 'center_x': 57.0,
    # 'r': 1,
    # 'w': 6,
    # 'y': 24,
    # 'x': 54
    # 'inners':   'num': 0,
    #             'contour': array([[[55, 29]], [[56, 28]], [[57, 28]],
    #                                 [[58, 29]], [[58, 31]], [[57, 32]],
    #                                 [[56, 32]], [[55, 31]]], dtype=int32),
    #             'center_y': 30.5,
    #             'c': 3,
    #             'inners': [],
    #             'h': 5,
    #             'center_x': 57.0,
    #             'r': 1,
    #             'w': 4,
    #             'y': 28,
    #             'x': 55}],

    if invalid:
        for pos, n in cx.load('numbers').items():
            if len(n['inners']) == 2:
                n['num'] = 8
            if len(n['inners']) == 1:
                i = n['inners'][0]
                # if pos == (4,5):
                #     print (i['y'] + i['h'], '<=', n['y'] + n['h']/2), (i['y'], '>=', n['y'] + n['h']/2)
                if i['center_y'] <= n['center_y']:
                    n['num'] = 9
                if i['center_y'] > n['center_y']:
                    n['num'] = 6



    # ==================================================================  Print Sudoku

    if invalid:
        table = np.zeros((9,9), np.uint8)
        for (r,c), n in cx.load('numbers').iteritems():
            table[r,c] = n['num']

        for r in range(0,9):
            for c in range(0,9):
                if table[r,c] == 0:
                    print "_",
                else:
                    print table[r,c],
                if (c+1) % 3 == 0:
                    sys.stdout.write(' |')
            sys.stdout.write("\n")
            if (r+1) % 3 == 0:
                print "------#------#------"
        cx.store('parsed', table)
        sys.stdout.flush()
    # ==================================================================  CHECK

    if invalid:
        ground_truth = np.array([
            [3, 4, 0, 6, 0, 0, 0, 5, 0],
            [2, 0, 0, 8, 1, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 4, 2, 6],
            [9, 0, 1, 0, 8, 2, 5, 0, 0],
            [6, 0, 0, 3, 0, 9, 0, 0, 7],
            [0, 0, 8, 1, 4, 0, 2, 0, 3],
            [8, 9, 6, 0, 0, 0, 7, 0, 0],
            [0, 0, 0, 0, 7, 5, 0, 0, 2],
            [0, 5, 0, 0, 0, 8, 0, 3, 1]
            ])
        parsed = cx.load('parsed')
        numbers = cx.load('numbers')
        for r in range(0,9):
            for c in range(0,9):
                if parsed[r,c] != ground_truth[r,c]:
                    pass
                    #print "Err in r", r, "c", c, " Got ", parsed[r,c], "instead of", ground_truth[r,c]




    print "Done"
    sys.stdout.flush()
    return cx.load('parsed')



img = load_image(path='assets/test/s1.jpg', w=550)
cx.add_buffer('orig', src=img)
#cx.add_buffer('laplace', shape=img.shape)
#cx.add_buffer('canny', shape=img.shape)
#cx.add_buffer('corners', shape=img.shape)
#cx.add_buffer('out', shape=img.shape)

sud = improve(src=cx.b('orig'))

cx.redraw()
cx.save_all_buffers()
cx.eventloop()
