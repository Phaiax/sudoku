import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('assets/test/s1.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img, (400, 400))

cv2.line(img,(0,0),img.shape[0:2],(255,0,0),5)

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball


# plt.imshow(img, interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
color = (0, 0, 0)

# mouse callback function
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,color

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),color,-1)
            else:
                cv2.circle(img,(x,y),5,color,-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),color,-1)
        else:
            cv2.circle(img,(x,y),5,color,-1)


def nothing(x):
    pass

# Create a black image, a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
        cv2.setTrackbarPos(switch, 'image', 1 if mode else 0)
    elif k == 27:
        break

    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    mode = s != 0

    color = (b,g,r)


cv2.destroyAllWindows()

