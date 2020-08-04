import cv2
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

image = cv2.imread('datalatih/apel3.jpg')

def grayscale(image):
    grayValue = 0.1140 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.2989 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


def cannyDetection(thresImage):
    edges = cv2.Canny(thresImage,25,255,L2gradient=False)  
    cv2.imwrite("hasil.bmp", edges )
    h,w = edges.shape
    # for i in range(h):
    #     for j in range(w):
    #         if edges[i,j] == 255:
    #             edges[i,j] = 1
    return edges


#start ekstraksi fitur GLCM 
def extraction_feature(img):
    img = Image.open(img).convert('L')
    pixelMap = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixelMap[i,j] == (i,j,255):
                pixelMap[i,j] = (i,j, 1)
            # if img[i,j] == 255:
            #     img[i,j] = 1
    mt = np.savetxt('hasilopen',np.array(img),fmt="%s")
    cv2.imwrite("hasilopen.bmp", np.array(img) )
    gl_0 = glcm(img,0)
    gl_45 = glcm(img,45)
    gl_90 = glcm(img,90)
    gl_135 = glcm(img,135)

    feature = np.array([np.average([contrast(gl_0),energy(gl_0),homogenity(gl_0),entropy(gl_0)]), np.average([contrast(gl_45),energy(gl_45),homogenity(gl_45),entropy(gl_45)]),
                       np.average([contrast(gl_90),energy(gl_90),homogenity(gl_90),entropy(gl_90)]), np.average([contrast(gl_135),energy(gl_135),homogenity(gl_135),entropy(gl_135)])])
    return feature

def contrast(matrix):
    width,height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += matrix[i][j] * np.power(i-j,2)
    return res 

def energy(matrix):
    width,height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += np.power(matrix[i][j],2)
    return res

def homogenity(matrix):
    width,height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += matrix[i][j] / (1 + np.power(i - j,2))   
    return res 

def entropy(matrix):
    width,height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            if matrix[i][j] > 0:
                res += matrix[i][j] * np.log2(matrix[i][j])
    return res

def glcm(img, degree):
    img = img.resize([128,128], Image.NEAREST)
    arr = np.array(img)
    res = np.zeros((arr.max() + 1, arr.max() + 1 ), dtype=int)
    width, height = arr.shape
    if degree == 0:
        for i in range(width - 1):
            for j in range(height):
                res[arr[j,i+1], arr[j,i]] += 1
    elif degree == 45:
        for i in range(width - 1):
            for j in range(1, height):
                res[arr[j-1, i+1], arr[j, i]] += 1
    elif degree == 90:
        for i in range(width):
            for j in range(1, height):
                res[arr[j-1, i], arr[j,i]] += 1
    elif degree == 135:
        for i in range(1, width):
            for j in range(1, height):
                res[arr[j-1,i-1], arr[j,i]] += 1
    else:
        print('sudut tidak valid')
    return res                                                    
#end ekstraksi fitur glcm



#start klasifikasi SVM

#end klasisfikasi SVM
                


imgGray = grayscale(image)
imgCanny = cannyDetection(imgGray)
imgFeatureExtraction = extraction_feature('hasil.bmp')


# print(imgCanny)
mt = np.savetxt('matrix',np.array(imgFeatureExtraction),fmt="%s")
# plt.figure()
# plt.imshow(imgCanny, cmap=plt.cm.gray)
# plt.title('-')
# plt.axis('off')
# plt.show()