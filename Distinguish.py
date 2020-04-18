import numpy as np
import imutils
import cv2


class ImageRecognition:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.config_path = {'one': './rabbit/one.jpg', 'two': './rabbit/two.jpg', 'three': './rabbit/three.jpg'}
        self.above_path = './above/above.jpg'
        self.matrix = []
        self.result = []
        self.image = None

    def recognition_picture(self):
        cropped = self.img[0:160, 0:320]
        cv2.imwrite(self.above_path, cropped)
        self.template_picture()

    def template_picture(self):
        one = self.img[160:180, 0:20]
        two = self.img[160:180, 25:50]
        three = self.img[160:180, 52:80]
        cv2.imwrite("./rabbit/one.jpg", one)
        cv2.imwrite("./rabbit/two.jpg", two)
        cv2.imwrite("./rabbit/three.jpg", three)
        self.direction()

    def rotate_bound_white_bg(self, image, angle):
        image = cv2.imread(image)
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        # borderValue 缺失背景填充色彩，此处为白色，可自定义
        return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
        # borderValue 缺省，默认是黑色（0, 0 , 0）
        # return cv2.warpAffine(image, M, (nW, nH))

    def direction(self):
        for i in self.config_path:
            # img = Image.open(self.config_path[i])
            # image = cv2.imread(self.config_path[i])
            # for j in range(1, 13):
            #     img2 = img.rotate(30 * j, expand=1)  # 自定义旋转度数
            #     img2.save("./rabbit/%s/%s.jpg" % (i, (i + str(j))))
            for angle in np.arange(0, 360, 15):
                rotated = self.rotate_bound_white_bg(self.config_path[i], angle)
                cv2.imwrite("./rabbit/%s/%s.jpg" % (i, (i + str(angle))), rotated)
        self.mark()

    def mark(self):
        for i in self.config_path:
            print(self.config_path[i], '<----------path')
            position = []
            for angle in np.arange(0, 360, 15):
                templatepath = "./rabbit/%s/%s.jpg" % (i, (i + str(angle)))
                # 读取模板图片
                template = cv2.imread(templatepath)
                # 转换为灰度图片
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                # 执行边缘检测
                template = cv2.Canny(template, 50, 200)
                (tH, tW) = template.shape[:2]
                # 显示模板
                # cv2.imshow("Template", template)
                self.image = cv2.imread(self.above_path)
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                found = None
                # 循环遍历不同的尺度
                for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                    # 根据尺度大小对输入图片进行裁剪
                    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
                    r = gray.shape[1] / float(resized.shape[1])

                    # 如果裁剪之后的图片小于模板的大小直接退出
                    if resized.shape[0] < tH or resized.shape[1] < tW:
                        break

                    # 首先进行边缘检测，然后执行模板检测，接着获取最小外接矩形
                    edged = cv2.Canny(resized, 50, 200)
                    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                    # 如果发现一个新的关联值则进行更新
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc, r)

                # 计算测试图片中模板所在的具体位置，即左上角和右下角的坐标值，并乘上对应的裁剪因子
                print(found[0])
                (_, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
                cv2.rectangle(self.image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # cv2.imshow("Image", self.image)
                # cv2.waitKey(2500)
                position.append({'found': found[0], 'coordinate': [startX, startY, endX, endY]})
            self.matrix.append(position)
        self.calculation()

    def calculation(self):
        for key, value in enumerate(self.matrix):
            cc = sorted(value, key=lambda x: x['found'], reverse=True)
            result_pciture = cv2.rectangle(self.image, (cc[0]['coordinate'][0], cc[0]['coordinate'][1]),
                                           (cc[0]['coordinate'][2], cc[0]['coordinate'][3]), (0, 0, 255), 2)

            cv2.imwrite("result_picture.jpg", result_pciture)
            a = np.array([cc[0]['coordinate'][0], cc[0]['coordinate'][1]])
            b = np.array([cc[0]['coordinate'][2], cc[0]['coordinate'][3]])
            xy = a + ((b - a) // 2)
            self.result.append([xy[0], xy[1]])
            # cv2.waitKey(2500)

    def start(self):
        self.recognition_picture()
        return self.result


test = ImageRecognition('661.jpg')
cc = test.start()
print(cc)
