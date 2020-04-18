import numpy as np
import imutils
import cv2
images = 'cc.jpg'
for i in range(1, 13):
    templatepath = 'three' + '%s.jpg' % str(i)
    # print(args)
    # 读取模板图片
    template = cv2.imread(templatepath)
    # 转换为灰度图片
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # 执行边缘检测
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    # 显示模板
    # cv2.imshow("Template", template)

    # 遍历所有的图片寻找模板
    # for imagePath in glob.glob(args["images"] + "/*.jpg"):
    # 读取测试图片并将其转化为灰度图片
    # print('test')
    image = cv2.imread(images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        # 结果可视化
        # if args.get("visualize", False):
        #     # 绘制矩形框并显示结果
        # clone = np.dstack([edged, edged, edged])
        # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        # cv2.imshow("Visualize", clone)
        # cv2.waitKey(0)

        # 如果发现一个新的关联值则进行更新
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # 计算测试图片中模板所在的具体位置，即左上角和右下角的坐标值，并乘上对应的裁剪因子
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # 绘制并显示结果
    print(found)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1500)