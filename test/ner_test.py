# 모듈 임포트
import os
import sys
from pathlib import Path

# if 'JAVA_HOME' not in os.environ:
#   os.environ["JAVA_HOME"] = JAVA_HOME

# # 프로젝트 루트 디렉토리를 파이썬 경로에 추가
# current_dir = Path(__file__).resolve().parent 
# root_dir = current_dir.parent
# sys.path.append(str(root_dir))

from config.GlobalParams import * 
root_dir = init_environment(__file__, up_levels=2)

# print(root_dir)

# 전처리 클래스
from utils.preprocess import Preprocess

# 모델 클래스 
from model.ner.ner_model import NerModel

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# ner 모델 호출
ner = NerModel(model_name=os.path.join(root_dir, 'model', 'ner', 'ner_model_1.keras'), preprocess=p)

# query = '내일 오전 10시에 탕수육 주문하고 싶어요.'
# query = '삼성전자 주식은 언제 오르나요?'
query = '일요일 12시 자장면 주문할께요'

predicts = ner.predict(query)
# print(predicts)

predict_tags = ner.predict_tags(query)
print(predict_tags)
