import numpy as np
#
# xy_f = [37, 36]
# xy_end = [75, 66]
#
# a = np.array(xy_f)
# b = np.array(xy_end)
# print(xy_f + ((b - a) // 2))
import imutils
import cv2
image = cv2.imread('./rabbit/one.jpg')
for angle in np.arange(0, 360, 15):
    print(angle)
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(1500)

