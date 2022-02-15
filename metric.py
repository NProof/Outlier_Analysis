# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

err_dir = Path("../dataset/test/")
data_fn = err_dir / "data.csv"
label_fn = err_dir / "eLabel.csv"
day_label_fn = err_dir / "day_Label.csv"

def example_001():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
if __name__ == "__main__":
    print("metric")
    
    # 讀取修改資料拆分的檔案
    # data_err = pd.read_csv(data_fn, index_col = 0)
    # label_err = pd.read_csv(label_fn, index_col = 0)
    labelD_err = pd.read_csv(day_label_fn, index_col = 0)["label"]
    
    print('{:.6f} ({}/{})'.format(
        sum(labelD_err)/len(labelD_err),
        sum(labelD_err),
        len(labelD_err)
    ))
    
    labelD_true = labelD_err
    labelD_predict = [
        bool(i) for i in np.random.randint(0, 2,size = len(labelD_err))
        ]
    
    cm = confusion_matrix(labelD_true, labelD_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    plt.show()
    