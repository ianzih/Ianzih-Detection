from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 從參數讀取圖檔路徑
files = os.listdir("./eval_img/")  # sys.argv[1:]

# 載入訓練好的模型
net = load_model('./logs/resnext101-082200.h5')

cls_list = [ 'Gray_black_uncertain','normal','white','yellow']

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory('./eval_img/',
                                                  target_size=(224, 224),
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  #batch_size=130,
                                                  shuffle=False)
epoch = 3
acc = []
# 辨識每一張圖
while epoch != 0:
    pred = net.evaluate(valid_batches,steps=100)
    print(pred[1])
    acc.append(pred[1])
    epoch -= 1
accuracy = np.mean(acc)
print(accuracy)

"""
prediction_classes = np.array([])
true_classes =  np.array([])

for x, y in valid_batches:
    prediction_classes = np.concatenate([prediction_classes,
                        np.argmax(net.predict(x), axis = -1)])
    true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])
print(classification_report(true_classes, prediction_classes))
"""