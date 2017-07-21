
# LOAD AND DISPLAY
# ==============================================

# imread -> numpy ndarray
img = cv2.imread(filename, cv2.IMREAD_COLOR|IMREAD_GRAYSCALE|IMREAD_UNCHANGED) # UNCHANGED includes alpha

# show with matplotlib
from matplotlib import pyplot as plt
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# with gtk
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


# empty
img = np.zeros((512,512,3), np.uint8)

# TYPE NDARRAY
# ===============================================

img.shape # -> tuple
img.ndim
img.dtype # for debugging! uint8

# img.T             img.base          img.ctypes        img.dumps         img.itemset
# img.nonzero       img.reshape       img.sort          img.tofile
# img.all           img.byteswap      img.cumprod       img.fill          img.itemsize
# img.partition     img.resize        img.squeeze       img.tolist
# img.any           img.choose        img.cumsum        img.flags         img.max
# img.prod          img.round         img.std           img.tostring
# img.argmax        img.clip          img.data          img.flat          img.mean
# img.ptp           img.searchsorted  img.strides       img.trace
# img.argmin        img.compress      img.diagonal      img.flatten       img.min
# img.put           img.setfield      img.sum           img.transpose
# img.argpartition  img.conj          img.dot           img.getfield      img.nbytes
# img.ravel         img.setflags      img.swapaxes      img.var
# img.argsort       img.conjugate     img.dtype         img.imag          img.ndim
# img.real          img.shape         img.take          img.view
# img.astype        img.copy          img.dump          img.item          img.newbyteorder
# img.repeat        img.size          img.tobytes

# data layout:
# [BGR]
px = img[100,100]
px_blue = img[100,100,0]
img[100,100] = [255,255,255]


# select region
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

# splitting channels
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))

# setting red to zero
img[:,:,2] = 0

# DRAW
# ===============================================
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> None
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# poly
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
# text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)


# INTERACTIVE
# ================================================

# Trackbars
cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar(switch, 'image',0,1,nothing) # switch = '0 : OFF \n1 : ON'
cv2.setTrackbarPos(switch, 'image', 1 if mode else 0)
s = cv2.getTrackbarPos(switch,'image')

# eventloop
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(timeout) & 0xFF
    if k == ord('m'):
        pass # key m pressed
    elif k == 27:
        break
    # getTrackbarPos()

# mouse
cv2.setMouseCallback('image',draw_circle)
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,color
    if event == cv2.EVENT_LBUTTONDOWN: # list of all events: [i for i in dir(cv2) if 'EVENT' in i]
        pass


# INTRESTING FUNCTIONS
# ==================================================

# Make special boarders around img (for kernel functions). [i for i in dir(cv2) if 'BORDER' in i]
cv2.copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> dst

cv2.cvtColor(src, code[, dst[, dstCn]]) -> dst

# THRESHOLDING
    # type=cv2.THRESH_BINARY | THRESH_BINARY_INV | THRESH_TRUNC | THRESH_TOZERO | THRESH_TOZERO_INV
    cv2.threshold(src, thresh, maxval, type[, dst]) -> retval, dst
    # automatically find best threshold
    cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) -> retval, dst
    #  adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
    cv2.inRange(src, lowerb, upperb[, dst]) -> dst

# BITWISE
    cv2.bitwise_not(src[, dst[, mask]]) -> dst
    cv2.bitwise_and(src1, src2[, dst[, mask]]) -> dst
    cv2.bitwise_or(src1, src2[, dst[, mask]]) -> dst
    cv2.bitwise_xor(src1, src2[, dst[, mask]]) -> dst

# BLUR
    # Averaging
    blur(src, ksize=(,) [, dst[, anchor[, borderType]]]) -> dst
    boxFilter(src, ddepth, ksize=(,) [, dst[, anchor[, normalize[, borderType]]]]) -> dst
    # Gaussian
    cv2.GaussianBlur(src, ksize=(,), sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
    # Median (does not introduce new color values)
    cv2.medianBlur(src, ksize[, dst]) -> dst
    # Bilateral (preserves edges)
    bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst


# KERNEL
    kernel = np.ones((5,5),np.float32)/25
    filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst

# MORPHOLOGY
    # Kernel: Rectangular
        kernel = np.ones((5,5),np.uint8)
        # or if not rectangular structured:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT | MORPH_ELLIPSE | MORPH_CROSS,(5,5))
    eroded = cv2.erode(img,kernel,iterations = 1)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# GRADIENTS / HIGH PASS FILTERS
    # if ksize=-1 -> Scharr is used
    cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst
    cv2.Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst
    cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

# PYDAMIDS
    # gaussian scale area by 1/4 or 4
    cv2.pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst # times 4 pixels
    cv2.pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst # times 1/4 pixels
    # Laplacian pyramid for level x
    cv2.substract(levelx, levelx_then_downscaled_then_upscaled)

# CONTOURS
    # finds white spots (best with b/w images, no grey)
    # mode = cv2.CHAIN_APPROX_NONE | cv2.CHAIN_APPROX_SIMPLE
    findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
    drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> None
    cv2.moments(contours[i])
    cv2.contourArea(contours[i]) -> area
        equi_diameter = np.sqrt(4*area/np.pi)
    cv2.arcLength(contours[i], closed)
    cv2.approxPolyDP(contours[i],epsilon,True)
    cv2.convexHull(contours[i] [, hull[, clockwise[, returnPoints]]) -> hull
    cv2.isContourConvex(contours[i])
    cv2.boundingRect(contours[i]) -> x,y,w,h
        aspect_ratio = float(w)/h
        extent = float(area)/(w*h)
        solidity = float(area)/cv2.contourArea(hull)
    cv2.minAreaRect(contours[i]) -> rect # rotated bounding rect
    cv2.minEnclosingCircle(contours[i])
    cv2.fitEllipse(contours[i])
    cv2.fitLine(contours[i], cv2.DIST_L2,0,0.01,0.01) -> [vx,vy,x,y]

# Histogram
    cv2.calcHist(images=[img], channels=[0], mask=None, histSize=[256], ranges=[0,256]
            [, hist[, accumulate]]) -> hist
    # draw
    plt.hist(img.ravel(),256,[0,256]); plt.show()
    # equalization
    equ = cv2.equalizeHist(img [,equ])
    # better localized equalization (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    # two dimensional hist
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist,interpolation = 'nearest')
    # Backprojection: Use histogram of searched object to get a probability mask from a picture
    cv2.calcBackProject
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_backprojection/py_histogram_backprojection.html


# Fourier Transformation
    # Speedup with optimal size, pad with zeros
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    nimg = cv2.copyMakeBorder(img, 0, nrows - rows, 0, ncols - cols, cv2.BORDER_CONSTANT, value = 0)
    #
    dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) # make 0,0 into center
    #                                            real plane       imag plane
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    # or
    cv2.cartToPolar(x, y[, magnitude[, angle[, angleInDegrees]]]) -> magnitude, angle
    # and inverse
    f_ishift = np.fft.ifftshift(fshift) # reverse np.fft.fftshift
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# Template matching
    methods = cv2.TM_CCOEFF | cv2.TM_CCOEFF_NORMED | cv2.TM_CCORR |
                cv2.TM_CCORR_NORMED | cv2.TM_SQDIFF | cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # or for multiple
    threshold = 0.8
    loc = np.where( res >= threshold)

# Hough transform
    # input is binary image (from canny)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    # see for display
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    # faster:
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# Hough transformation for circles
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

# FOREGROUND/BACKGROUND SEPERATION
    # wathershed and playing with erosion/dillution to seperate fore and background
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

    # Grab cut
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html


# Calculates the distance to the closest zero pixel for each pixel of the source image.
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2|DIST_L1|DIST_C, maskSize=5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.minMaxLoc(imgray,mask) -> min_val, max_val, min_loc, max_loc
cv2.mean(im,mask) -> mean_val
cv2.countNonZero(img)

# IMAGE MANIPILATION
# ==================================================

# MERGING
    cv2.add(img1, img2) # is saturating
    cv2.addWeighted(img1,a,img2,b,c) # a*img1 + b*img2 + c

# AFFINE TRANSFORM
    # In affine transformation, all parallel lines in the original image will still be parallel in the output image.
    warpAffine(src, M, dsize=(,) [, dst[, flags[, borderMode[, borderValue]]]]) -> dst
    # TRANSLATE
    M = np.float32([[1,0,tx],[0,1,ty]])
    # ROTATE
    M = getRotationMatrix2D(center=(,), angle, scale=(,))
    # FROM POINTS
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)

# PERSPECTIVE TRANSFORM
    #
    warpPerspective(src, M, dsize=(,) [, dst[, flags[, borderMode[, borderValue]]]]) -> dst
    # FROM POINTS
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)




# PATTERNS
# ==================================================

# MASKING AND ADDING
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst

# PERFORMANCE
# ==================================================
e1 = cv2.getTickCount()
# your code execution
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency() # time is in seconds
cv2.useOptimized() # is using optimized?
cv2.setUseOptimized(True|False)

# ipython
%timeit c=d()
