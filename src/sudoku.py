import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from context import Context


# GLOBALS
cx = Context()
adaptive_kernel_size = 17
redraw = None

def load_image(path, w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h = int( img.shape[1] * w / img.shape[0] )
    return cv2.resize(img, (w, h))



def on_trackbar_change(x):
    global cx
    improve(src=cx.b('orig'))
    cx.redraw()


def improve(src):
    global cx
    #(a, _) = cx.get_slider('a',on_trackbar_change)
    #k = int(a/8*2+3)
    #(m, _) = cx.get_toggle('m', 1, callback=on_trackbar_change)
    #(n, _) = cx.get_toggle('n', 1, callback=on_trackbar_change)


    cv2.Laplacian(src, ddepth=cv2.CV_8U, ksize=3, dst=cx.b('laplace'))

    print "Done"
    sys.stdout.flush()


img = load_image(path='assets/test/s1.jpg', w=700)
cx.add_buffer('orig', src=img)
cx.add_buffer('laplace', shape=img.shape)
#cx.add_buffer('laplace', shape=img.shape)
#cx.add_buffer('laplace', shape=img.shape)

improve(src=cx.b('orig'))
cx.redraw()
cx.eventloop()
