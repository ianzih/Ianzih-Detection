from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import os
import cv2
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 從參數讀取圖檔路徑
#files = os.listdir("./test_img/non-normal")  # sys.argv[1:]

# 載入訓練好的模型
net = load_model('./logs/model-resnext101-final1e-06.h5')

cls_list = [ 'Gray_black_uncertain','normal','white','yellow']

test = "normal"
# 辨識每一張圖
with open(test + ".txt", 'w') as intxt:
    lst=[str(test), '\n\n']
    intxt.writelines(lst)
    for f in os.listdir("./test_img/" + test):
        #img = cv2.imread("./test_img/normal" + "/" + f, target_size=(224, 224))
        img = image.load_img("./test_img/" + test + "/" + f, target_size=(224, 224))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        print(f)
        intxt.write(str(f))
        intxt.write('\n')
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
            lst = ['    {:.3f}  {}'.format(pred[i], cls_list[i]),'\n']
            intxt.writelines(lst)
        intxt.write('\n')