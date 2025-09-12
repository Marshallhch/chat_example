import os
import sys
from pathlib import Path

import tensorflow as tf

if 'JAVA_HOME' not in os.environ:
  os.environ["JAVA_HOME"] = JAVA_HOME

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent 
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from config.GlobalParams import * 

# 커스텀 모듈
from utils.preprocess import Preprocess
from model.intent.intent_model import IntenModel

# 소프트맥스 업그레이드 버전 사용
custom_objects = {'softmax_v2': tf.nn.softmax}

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# 의도 분류 모델 호출
intent = IntenModel(model_name=os.path.join(root_dir, 'model', 'intent', 'intent_model.keras'), preprocess=p, custom_objects=custom_objects)

# 테스트 질문
query = '내일 오전 10시에 탕수육 주문하고 싶어요.'

# 의도 클래스 예측
predict = intent.predict_class(query)