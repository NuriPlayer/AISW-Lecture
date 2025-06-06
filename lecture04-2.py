import numpy as np;
import pandas as pd;
import tensorflow as tf;

try:
    df = pd.read_csv('./data/02_Exam_dataset.csv')
except Exception as e:
    print("파일 불러오기 실패", e)
    
