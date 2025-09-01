import cv2
# import numpy as np


# img = cv2.imread("epoch_45.png")
img2 = cv2.imread("pawan.jpeg")
# #Accessing Pixels
# # (b,g,r) = img[100,50]
# # img[100,50] = (255,255,0) #change pixels color
# # Show the image in a window

# # print(img.shape)   # (height, width, channels)
# # print(img.size)    # total pixels
# # print(img.dtype)   # data type (usually uint8)


# cv2.imshow("Image Window", img)

# cv2.waitKey(0)       # wait for any key press
# cv2.destroyAllWindows()

# # save image
# # cv2.imwrite("output.png",img)


# import numpy as np


# Drawing on an Image 

# x = np.zeros((500,500,3),dtype='uint8')
#cv2.line(image,start,end,color,thickness)
# cv2.line(img,(20,50),(50,20),(255,0,255),2)
# cv2.rectangle(img,(40,100),(80,150),(0,0,0),-1)

# # cv2.circle(image, center, radius, color, thickness)
# cv2.circle(img,(10,10),50,(0,0,255),1)
# cv2.putText(img,"Hey ! Pawan",(50,480),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
# cv2.imshow("Drawing", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Image Transformations

# Resize:
# small = cv2.resize(img2,(200,200))

# Resize with scale factor
# bigger = cv2.resize(img2,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)


'''   cv2.resize(src, dsize, fx, fy, interpolation

src → source image (img in your case).

dsize → desired size (width, height) in pixels.

Example: (200, 300) → makes image 200px wide and 300px tall.

If None, then OpenCV uses fx and fy scaling factors instead.

fx → scale factor along the x-axis (width).

1.5 → width is 1.5 times bigger.

fy → scale factor along the y-axis (height).

1.5 → height is 1.5 times bigger.

interpolation → method to compute new pixel values.

cv2.INTER_NEAREST → very fast, low quality.

cv2.INTER_LINEAR → good for enlarging (default).

cv2.INTER_CUBIC → slower but higher quality (best for upscaling).

cv2.INTER_LANCZOS4 → best for large enlargements.'''

# cv2.imshow("image",img2)
# cv2.imshow("small",small)
# cv2.imshow("big",bigger)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Crop an Image 

'''Images in OpenCV are arrays: img[rows, cols]

Rows = y-axis (height)

Cols = x-axis (width)'''

# crop = img2[100:1200,100:700]
# cv2.imshow("Crop",crop)
# cv2.imshow("real",img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Rotation. 

# (h,w) = img2.shape[:2]
# center = (w//2,h//2)

# # cv2.getRotationMatrix2D(center,angle,scale)
# M = cv2.getRotationMatrix2D(center,90,0.5)
# rotate = cv2.warpAffine(img2,M,(w,h))


# cv2.imshow("Original", img2)
# cv2.imshow("Rotated 45 Degrees", rotate)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''Rotate Without Cropping (Keeps Whole Image)'''

# import cv2
# import numpy as np

# Read image
# img = cv2.imread("pawan.jpeg")

# (h, w) = img.shape[:2]
# center = (w // 2, h // 2)

# Angle
# angle = 120

# # Get rotation matrix
# M = cv2.getRotationMatrix2D(center, angle, 1.0)

# # --- Step 1: Compute new bounding dimensions ---
# cos = np.abs(M[0, 0])
# sin = np.abs(M[0, 1])

# new_w = int((h * sin) + (w * cos))
# new_h = int((h * cos) + (w * sin))

# # --- Step 2: Adjust rotation matrix for translation ---
# M[0, 2] += (new_w / 2) - center[0]
# M[1, 2] += (new_h / 2) - center[1]

# # --- Step 3: Perform rotation with new size ---
# rotated = cv2.warpAffine(img, M, (new_w, new_h))

# cv2.imshow("Original", img)
# cv2.imshow("Rotated (No Crop)", rotated)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Mirror Images 
# Horizontal flip (mirror effect)
# mirror_h = cv2.flip(img2, 1)

# # Vertical flip
# mirror_v = cv2.flip(img2, 0)

# # Both flips
# mirror_both = cv2.flip(img2, -1)

# cv2.imshow("Original", img2)
# cv2.imshow("horizontal",mirror_h)
# cv2.imshow('vertical',mirror_v)
# cv2.imshow("both",mirror_both)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Color Conversion

'''
**  Why Color Spaces Matter

When an image is loaded in OpenCV (cv2.imread), it's in BGR format by default (not RGB!).
But AI models, libraries like TensorFlow/PyTorch, and even matplotlib expect RGB.
Also, sometimes we don't need full color at all — grayscale or HSV can simplify things.

'''

# BGR to RGB
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#GrayScale
# img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

'''
HSV (Hue, Saturation, Value)
Hue = color (0-180 in OpenCV)
Saturation = intensity of color
Value = brightness
Great for object detection by color (e.g., tracking a red ball).


LAB (Lightness, A, B)
L = lightness, A/B = color opponent channels.
Used in image enhancement and style transfer.
Convert using:

lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)'''

''' detect red color in image
# cv2.line(img2,(10,500),(50,200),(0,0,255),2)
# # hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# # lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

# hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# # Range 1: lower reds (0–10)
# lower_red1 = np.array([0, 120, 70])
# upper_red1 = np.array([10, 255, 255])
# mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

# # Range 2: upper reds (170–180)
# lower_red2 = np.array([170, 120, 70])
# upper_red2 = np.array([180, 255, 255])
# mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# # Combine both
# mask = mask1 | mask2
# result = cv2.bitwise_and(img2, img2, mask=mask)
'''

# cv2.imshow("Pawan",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Convert to LAB
# lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

# # Split into channels
# L, A, B = cv2.split(lab)

# # --- Adjust brightness here ---
# # Increase brightness by adding value (clip to 255)
# brightness = 40   # try positive (brighten) or negative (darken)
# L = np.clip(L + brightness, 0, 255).astype(np.uint8)

# # Merge back
# lab = cv2.merge([L, A, B])

# # Convert back to BGR
# bright_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# cv2.imshow("Original", img2)
# cv2.imshow("Brightened", bright_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# blur = cv2.GaussianBlur(img2, (5,5), 0) #odd numbers 5,7,9 with same shape
# blur = cv2.medianBlur(img2, 7)
# blur = cv2.bilateralFilter(img2, 9, 75, 75)

# blur1=cv2.GaussianBlur(img2, (9,9), 0)
# blur2=cv2.bilateralFilter(img2, 9, 75, 75)
# Gaussian makes the whole face “foggy,” while bilateral smooths skin but keeps sharp edges (eyes, lips, hair)

# cv2.imshow("o",img2)

# cv2.imshow("b",blur1)
# cv2.imshow('e',blur2)

# Example: sharpening kernel
# kernel = np.array([[0, -1,  0],
#                    [-1,  5, -1],
#                    [0, -1,  0]])

# sharpened = cv2.filter2D(img2, -1, kernel)
# cv2.imshow("sharpened" ,sharpened)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''Thresholding & Binarization'''
import cv2
import numpy as np 

# img = cv2.imread("image.png")

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# kernal = np.ones((3,3),np.uint8)
# cleaned = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernal)
# cv2.imshow("Original",img)
# cv2.imshow("Thresh",thresh)
# cv2.imshow("clean",cleaned)

'''
1. Contours & Object Detection

cv2.findContours → finds the outlines of shapes in a binary mask.
cv2.drawContours → draws them back on the image.
cv2.boundingRect → gives an axis-aligned box (rectangle).
cv2.minAreaRect → gives the smallest rotated rectangle that covers the object.

👉 Used in AI for: detecting objects after segmentation, finding blobs, counting objects (cells, cars, etc.).'''

# cv2.findContours(img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# video capturing

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

'''Codec: A way to compress video.

'mp4v' → safe codec for .mp4 files.

cv2.VideoWriter_fourcc(*'mp4v') → expands string into chars 'm' 'p' '4' 'v'.

cv2.VideoWriter(filename, fourcc, fps, frame_size)

'output.mp4' → output file name.

20.0 → frames per second.

(frame_width, frame_height) → video resolution.'''

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc,20.0,(frame_width,frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame,"hello Guys",(50,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    out.write(frame)

    cv2.imshow("frame ",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()