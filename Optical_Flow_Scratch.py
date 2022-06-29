import math
from scipy import signal
from PIL import Image
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf



def LK_OpticalFlow(Image1, Image2):


        # Frame 1

    I1 = np.array(Image1)
    I2 = np.array(Image2)
    S = np.shape(I1)

    # applying Gaussian filter of size 3x3 to eliminate any noise

    I1_smooth = cv2.GaussianBlur(I1, (3, 3), 0)  # input image
                                                 # shape of the kernel
                                                 # lambda
    I2_smooth = cv2.GaussianBlur(I2, (3, 3), 0)

    # First Derivative in X direction

    Ix = signal.convolve2d(I1_smooth, [[-0.25, 0.25], [-0.25, 0.25]],
                           'same') + signal.convolve2d(I2_smooth,
            [[-0.25, 0.25], [-0.25, 0.25]], 'same')

    # First Derivative in Y direction

    Iy = signal.convolve2d(I1_smooth, [[-0.25, -0.25], [0.25, 0.25]],
                           'same') + signal.convolve2d(I2_smooth,
            [[-0.25, -0.25], [0.25, 0.25]], 'same')

    # First Derivative in XY direction

    It = signal.convolve2d(I1_smooth, [[0.25, 0.25], [0.25, 0.25]],
                           'same') + signal.convolve2d(I2_smooth,
            [[-0.25, -0.25], [-0.25, -0.25]], 'same')

    # finding the good features

    features = cv2.goodFeaturesToTrack(I1_smooth, 10000, 0.01, 10)  # Input image
                                                                    # max corners
                                                                    # lambda 1 (quality)
                                                                    # lambda 2 (quality)

    feature = np.int0(features)
    fig = plt.figure()

    for i in feature:
        (x, y) = i.ravel()
        cv2.circle(I1_smooth, (x, y), 3, 0, -1)  # input image
                                                 # centre
                                                 # radius
                                                 # color of the circle
                                                 # thickness of the outline

    # creating the u and v vector

    u = v = np.nan * np.ones(S)

    # Calculating the u and v arrays for the good features obtained n the previous step.

    for l in feature:
        (j, i) = l.ravel()

        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.

        IX = [  # The x-component of the gradient vector
            Ix[i - 1, j - 1],
            Ix[i, j - 1],
            Ix[i - 1, j - 1],
            Ix[i - 1, j],
            Ix[i, j],
            Ix[i + 1, j],
            Ix[i - 1, j + 1],
            Ix[i, j + 1],
            Ix[i + 1, j - 1],
            ]
        IY = [  # The Y-component of the gradient vector
            Iy[i - 1, j - 1],
            Iy[i, j - 1],
            Iy[i - 1, j - 1],
            Iy[i - 1, j],
            Iy[i, j],
            Iy[i + 1, j],
            Iy[i - 1, j + 1],
            Iy[i, j + 1],
            Iy[i + 1, j - 1],
            ]
        IT = [  # The XY-component of the gradient vector
            It[i - 1, j - 1],
            It[i, j - 1],
            It[i - 1, j - 1],
            It[i - 1, j],
            It[i, j],
            It[i + 1, j],
            It[i - 1, j + 1],
            It[i, j + 1],
            It[i + 1, j - 1],
            ]

        # Using the minimum least squares solution approach

        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK))  # transpose of A
        LK = np.array(np.matrix.transpose(LK))

        A1 = np.dot(LK_T, LK)  # Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2, LK_T)

        (u[i, j], v[i, j]) = np.dot(A3, IT)  # we have the vectors with minimized square error


    c = 'b'

    # ======= Plotting the vectors on the image========


    plt.title('Vector plot of Optical Flow of good features')
    plt.imshow(I1, cmap=cm.gray)
    for i in range(S[0]):
        for j in range(S[1]):
            if abs(u[i, j]) > t or abs(v[i, j]) > t:  # setting the threshold to plot the vectors
                plt.arrow(
                    j,
                    i,
                    v[i, j],
                    u[i, j],
                    head_width=5,
                    head_length=5,
                    color=c,
                    )
    #fig.canvas.draw()


    # put pixel buffer in numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)


    cv2.imshow("plot", mat)
    #print(type(img))
    #print(img)
    #cv2.imshow('l' , np.array(img, dtype = np.uint8 ) )











t = 0.1  # choose threshold value
fig = plt.figure()
# import the images
viewer = fig.add_subplot(111)
plt.ion()  # Turns interactive mode on (probably unnecessary)
fig.show()  # Initially shows the figure

#video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc('A','V','C','1'), 1, (mat.shape[0],mat.shape[1]))
cap = cv2.VideoCapture("Vid_Sample2.mp4")

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prevgray = cv2.resize(prevgray, (640,480), interpolation = cv2.INTER_AREA)
while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)

    # start time to calculate FPS
    start = time.time()

    mat = LK_OpticalFlow(gray, prevgray)

    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")

    #cv2.imshow('flow HSV', mat)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
