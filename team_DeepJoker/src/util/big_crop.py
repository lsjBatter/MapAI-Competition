import numpy as np
import math
from torchvision import transforms as T
import cv2
import os
from torchvision import transforms as transforms
def label_to_onehot(label,colorDict_GRAY,classNum=8):
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    new_label = np.zeros(label.shape + (classNum,))
        #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i,i] = 1                                          
    label = new_label
    return label   
#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1 ,colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY
#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength,path_size):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (path_size - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (path_size - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (path_size - SideLength * 2) : i * (path_size - SideLength * 2) + path_size,
                          j * (path_size - SideLength * 2) : j * (path_size - SideLength * 2) + path_size]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (path_size - SideLength * 2) : i * (path_size - SideLength * 2) + path_size,
                      (img.shape[1] - path_size) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - path_size) : img.shape[0],
                      j * (path_size-SideLength*2) : j * (path_size - SideLength * 2) + path_size]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - path_size) : img.shape[0],
                  (img.shape[1] - path_size) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (path_size - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (path_size - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver,path_size):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i,img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : path_size - RepetitiveLength, 0 : path_size-RepetitiveLength] = img[0 : path_size - RepetitiveLength, 0 : path_size - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 512 - RepetitiveLength] = img[0 : ColumnOver, 0 : 512 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : path_size - RepetitiveLength] = img[path_size - ColumnOver - RepetitiveLength : path_size, 0 : path_size - RepetitiveLength]
            else:
                result[j * (path_size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (path_size - 2 * RepetitiveLength) + RepetitiveLength,
                       0:path_size-RepetitiveLength] = img[RepetitiveLength : path_size - RepetitiveLength, 0 : path_size - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : path_size - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : path_size - RepetitiveLength, path_size -  RowOver: path_size]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[path_size - ColumnOver : path_size, path_size - RowOver : path_size]
            else:
                result[j * (path_size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (path_size - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : path_size - RepetitiveLength, path_size - RowOver : path_size]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : path_size - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (path_size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (path_size - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : path_size - RepetitiveLength, RepetitiveLength : path_size - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (path_size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (path_size - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[path_size - ColumnOver : path_size, RepetitiveLength : path_size - RepetitiveLength]
            else:
                result[j * (path_size - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (path_size - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (path_size - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (path_size - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : path_size - RepetitiveLength, RepetitiveLength : path_size - RepetitiveLength]
    return result
if __name__ == "__main__":
    path_size = 512
    area_perc = 0.5
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * path_size / 2)
    img = cv2.imread('/boot/data1/kang_data/Inria/test/austin5.tif',-1)
    ImgArray, RowOver, ColumnOver = TifCroppingArray(img, RepetitiveLength,path_size)
    print(len(ImgArray))
    print(len(ImgArray[0]))
    # transform = transforms.Compose([
    #     transforms.ToTensor(), 
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # path = []
    # for i in range(len(ImgArray)):
    #     for j in range(len(ImgArray[0])):
    #         path_image = ImgArray[i][j]
    #         path_image = transform(path_image)
    #         path.append(path_image)
    # print(len(path))