import cv2

img = cv2.imread("222.jpg")
cropped = img[0:160, 0:320]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("cc.jpg", cropped)
img = cv2.imread("222.jpg")
one = img[160:180, 0:25]  # 裁剪坐标为[y0:y1, x0:x1]
two = img[160:180, 25:50]  # 裁剪坐标为[y0:y1, x0:x1]
three = img[160:180, 52:80]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("one.jpg", one)
cv2.imwrite("two.jpg", two)
cv2.imwrite("three.jpg", three)
from PIL import Image
img = Image.open("three.jpg")
for i in range(1, 13):
    img2 = img.rotate(30 * i)  # 自定义旋转度数
    img2.save("three%s.jpg" % str(i))

