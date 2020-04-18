from PIL import Image

img = Image.open("test.png")

# # 旋转方式一
# # print(Image.ROTATE_90)
# for i in range(0, 3):
#     img1 = img.transpose(i + 2)
#     # mypath = './picture/%swoman%s.png' % str(i)
#     # print(mypath)
#     img1.save('./picture/woman%s.png' % str(i))
# img1 = img.transpose(Image.ROTATE_90)   # 引用固定的常量值
# img1.save("./picture/headw3.png")

# # 旋转方式二
img2 = img.rotate(-50)  # 自定义旋转度数

# img2 = img2.resize((400, 400))   # 改变图片尺寸
img2.save("r2.png")
