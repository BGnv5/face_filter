import cv2
import matplotlib.pyplot as plt
import numpy as np

"""##### 1.读取图片 """
img = cv2.imread('./face.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_filter)
# plt.show()

"""##### 2.直接对图像进行滤波 """
# kernel_size = 3  # 效果不明显
kernel_size = 17  # 太模糊了
kernel = np.ones((kernel_size,kernel_size), np.float32) / (kernel_size*kernel_size)
img_filter = cv2.filter2D(img_rgb, -1, kernel)
# plt.imshow(cv2.hconcat([img_rgb, img_filter]))
# plt.show()

'''##### 3.肤色检测，只对肤色区域进行滤波 '''
result = img_filter.copy()
# 把图像转换到HSV色域
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 对图像进行分割，分别获取h,s,v通道分量图像
(_h, _s, _v) = cv2.split(img_hsv)
# 根据源图像的大小创建一个全0的矩阵，用于保存图像数据
skin = np.zeros(_h.shape, dtype=np.float32)
# 获取源图像数据的长和宽
(height, weight) = _h.shape
# 遍历图像，判断HSV通道的数值，如果在指定范围中，则把新图像的点设为255，否则设为0
for i in range(height):
    for j in range(weight):
        if _h[i][j] > 5 and _h[i][j] < 120 and _s[i][j] > 18 and _s[i][j] < 255 and _v[i][j] > 50 and _v[i][j] < 255:
            skin[i][j] = 1.0
        else:
            skin[i][j] = 0.0
            result[i][j] = img_rgb[i][j]

plt.imshow(cv2.hconcat([img_rgb, img_filter, result]))
# plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
plt.show()




