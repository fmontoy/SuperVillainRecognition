from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import video

car_image_test = imread("/home/felipe/Universidad/SuperVillain/SuperVillainRecognition/Dataset/Originales/SNV248.JPG", as_grey=True)
car_image = frame
# it should be a 2 dimensional array
print(car_image.shape)
#/home/felipe/Universidad/SuperVillain/SuperVillainRecognition/Dataset/Originales/IMG_20180709_173912 (1).jpg
# the next line is not compulsory however, a grey scale pixel
# in skimage ranges between 0 & 1. multiplying it with 255
# will make it range between 0 & 255 (something we can relate better with

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()