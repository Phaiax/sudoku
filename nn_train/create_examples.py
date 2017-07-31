from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from six.moves import cPickle as pickle

#from context import Context
#cx = Context()


# get list of fonts
import subprocess
convert_fonts = subprocess.Popen(['convert', '-list', 'font'],
                          stdout=subprocess.PIPE).communicate()[0]

convert_fonts = convert_fonts.split()
fonts_paths = []
for i in range(0, len(convert_fonts)):
    if convert_fonts[i] == "glyphs:" and convert_fonts[i+1].endswith('.ttf'):
        fonts_paths.append(convert_fonts[i+1])



exclude = ['cmex10.ttf', 'cmsy10.ttf', 'esint10.ttf', 'fontawesome-webfont.ttf', 'KacstArt.ttf',
    'KacstBook.ttf', 'KacstDecorative.ttf', 'KacstDigital.ttf', 'KacstFarsi.ttf', 'KacstLetter.ttf',
    'KacstNaskh.ttf', 'KacstOffice.ttf', 'KacstPen.ttf', 'KacstPoster.ttf', 'KacstQurn.ttf',
    'KacstScreen.ttf', 'KacstTitle.ttf', 'KacstTitleL.ttf', 'lklug.ttf', 'mry_KacstQurn.ttf',
    'msam10.ttf', 'msbm10.ttf', 'opens___.ttf', 'rsfs10.ttf', 'stmary10.ttf', 'wasy10.ttf',
    'Padauk-Bold.ttf', 'eeyek.ttf', 'Lohit-Odia.ttf', 'NotoSansLisu-Regular.ttf',
    'NotoSansMandaic-Regular.ttf', 'NotoSansMeeteiMayek-Regular.ttf', 'NotoSansNKo-Regular.ttf',
    'NotoSansTagalog-Regular.ttf', 'NotoSansTaiTham-Regular.ttf', 'NotoSansTaiViet-Regular.ttf',
    'unifont_upper.ttf', 'NotoEmoji-Regular.ttf']

fonts_paths = [f for f in fonts_paths if f.split('/')[-1] not in exclude]

fonts = []
for f in fonts_paths:
    try:
        fonts.append([f, ImageFont.truetype(f, 40)])
    except:
        print "Error with:", f
#for i in [11,14,36,38,55,56,57,58,59,60,61,62,65,66,67,68,69,70,117,123,124,125,138,149,155,198]:
#    print "'" + fonts_paths[i].split('/')[-1] + "',"

img = Image.new("L", (300, 10000), "black")
draw = ImageDraw.Draw(img)

for i in range(0, len(fonts)):
    #print i, fonts[i][0]
    draw.text((0, (i*40)%10000 ), str(i) + " 123456789",(255),font=fonts[i][1])

img.save('src/sample-out.jpg')

#sys.exit()

fonts = fonts

w_canvas = 60
bg_color = 200
fg_color = 40
w = 60
w23 = 2*(w/3)
mismatch_max = w/25
pixel_depth = 255.0

num_distortions = 100
total_examples = len(fonts) * 9 * num_distortions
dl_size = (28,14)

print "examples:", total_examples
print "mem:", total_examples*dl_size[0]*dl_size[1]*4/1024/1024, "MB"

dl_dataset = np.zeros((total_examples,dl_size[0],dl_size[1]), dtype=np.float32)
dl_labels = np.zeros((total_examples), dtype=np.float32)
dl_index = 0

for f in fonts:
    for i in range(1,10):
        #i=3
        img = Image.new("L", (w_canvas, w_canvas), (bg_color))
        draw = ImageDraw.Draw(img)
        draw.text((mismatch_max, mismatch_max), str(i), fg_color, font=f[1])
        img = np.array(img) # convert to opencv
        #cx['i1'] = img

        # for later cutting
        _, thre_cp = cv2.threshold(img, (bg_color+fg_color)/2, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2,2),np.uint8)
        thre_cp = cv2.dilate(thre_cp,kernel,iterations = 1)
        _, contours, _ = cv2.findContours(thre_cp, mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)

        # 100 different distortions
        for _ in range(0, num_distortions):
            out_contour = np.random.random_sample((4, 2)) * (w/3)
            out_contour = np.array(out_contour, dtype=np.float32) \
                         + np.array([[0,0],[w23,0],[w23,w23],[0,w23]], dtype=np.float32)

            in_contour = np.array([[0,0],[w_canvas,0],[w_canvas,w_canvas],[0,w_canvas]], dtype=np.float32)

            M = cv2.getPerspectiveTransform(in_contour,out_contour)
            distorted = cv2.warpPerspective(img, M,
                dsize=(w,w), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            #cx['d1'] = distorted

            im = np.empty((w,w), np.uint8)
            cv2.randn(im, 30, 10)

            distorted = cv2.add(distorted,im)
            #cx['s1_raw'] = distorted
            #distorted = cv2.GaussianBlur(distorted, ksize=(3,3), sigmaX=1)
            distorted = cv2.blur(distorted, ksize=(3,3))
            #cx['s1_raw_blur'] = distorted

            # miss transformation
            in_contour =  in_contour + np.array((np.random.random_sample((4, 2)) - 0.5) * mismatch_max,
                                                dtype=np.float32)
            M = cv2.getPerspectiveTransform(out_contour, in_contour)
            back = cv2.warpPerspective(distorted, M,
                dsize=(w_canvas,w_canvas), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            #cx['back'] = back

            # improve back to normal
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equihist = clahe.apply(back)
            sudoku = cv2.fastNlMeansDenoising(equihist, h=16,
                templateWindowSize=7, searchWindowSize=10)
            sudoku = clahe.apply(sudoku)
            #cx['s1'] = sudoku

            thresholded = cv2.adaptiveThreshold(sudoku, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=20)
            # cx['s1_thresholded'] = thresholded

            #kernel = np.ones((3,3),np.uint8)
            #thre_cp = cv2.dilate(cx['s1_thresholded'],kernel,iterations = 1)
            #contours, _ = cv2.findContours(thre_cp, mode=cv2.RETR_TREE,
            #    method=cv2.CHAIN_APPROX_SIMPLE)

            #if len(contours) == 0:
            #    continue

            #with_area = [(c, cv2.contourArea(c)) for c in contours]
            #with_area.sort(key=lambda ca: ca[1])
            #largest = with_area[-1][0]
            largest = contours[0]

            bx,by,bw,bh = cv2.boundingRect(largest)
            area = bw*bh
            if area < 10:
                continue


            if bw < bh/2:
                bw = bh/2

            #for_dl = np.zeros(dl_size, dl_size))
            view = thresholded[by:by+bh,bx:bx+bw]
            for_dl = cv2.resize(view, dsize=(dl_size[1], dl_size[0]))

            #cx['for_dl'] = for_dl

            image_data = (for_dl.astype(float) - pixel_depth / 2) / pixel_depth
            dl_dataset[dl_index, :, :] = image_data
            dl_labels[dl_index] = i-1
            dl_index += 1

        print ".",
        sys.stdout.flush()
    #         break
    #     break
    print "\nP", sys.argv[-1], ":" , dl_index, "/", total_examples, "=", 100*dl_index/total_examples, "%",
    sys.stdout.flush()
    # break
dl_dataset = dl_dataset[0:dl_index,:,:]
dl_labels = dl_labels[0:dl_index]

pickle_file = "training_data_" + sys.argv[-1]
print "\nsave to", pickle_file, "...",
sys.stdout.flush()
with open(pickle_file, 'wb') as f:
    pickle.dump({'dataset': dl_dataset, 'labels': dl_labels}, f, pickle.HIGHEST_PROTOCOL)
print "done."
sys.stdout.flush()

#cx.redraw()
#cx.eventloop()

# draw 25 random
