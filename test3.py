from skimage.measure import compare_ssim
import cv2


class CompareImage():

    def compare_image(self, path_image1, path_image2):
        imageA = cv2.imread(path_image1)
        imageB = cv2.imread(path_image2)

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        print("SSIM: {}".format(score))
        return score


compare_image = CompareImage()
for i in range(0, 3):
    img1 = './picture/head2.png'
    img2 = './picture/%s.png' % ('headw' + str(i))
    print(img2)
    compare_image.compare_image(img1, img2)
