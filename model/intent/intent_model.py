import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

if 'JAVA_HOME' not in os.environ:
  os.environ["JAVA_HOME"] = JAVA_HOME

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent 
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from config.GlobalParams import * 

# 소프트맥스 업그레이드 버전 사용
custom_objects = {'softmax_v2': tf.nn.softmax}

# 의도 분류 모듈 클래스
class IntenModel:
  def __init__(self, model_name, preprocess, custom_objects=None):
    # 의도 클래스 레이블
    self.labels = {0: '인사', 1: '욕설', 2: '주문', 3: '예약', 4: '기타'}
    # 의도 분류 모델 호출
    self.model = load_model(model_name, custom_objects=custom_objects)
    # 전처리 객체
    self.p = preprocess

  # 의도 클래스 예측
  def predict_class(self, query):
    pass