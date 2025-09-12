import os
import sys
import pickle
from pathlib import Path

if 'JAVA_HOME' not in os.environ:
  os.environ["JAVA_HOME"] = JAVA_HOME

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent # 현재 경로
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from config.GlobalParams import *
from utils.preprocess import Preprocess
from tensorflow.keras import preprocessing

# 말뭉치 데이터 파일 절대 경로 
corpus_dict_path = os.path.join(root_dir, 'train_tools', 'dict', 'corpus.txt')

# 말뭉치 데이터 문자열 편집 함수
def read_corpus_data(filename):
  with open(filename, 'r', encoding='utf-8') as f:
    data = [line.strip().split('\t') for line in f.read().splitlines()]
  return data

# 편집 데이터 확인
corpus_data = read_corpus_data(corpus_dict_path)
# print(corpus_data[:5])

# 말뭉치에서 키워드만 추출하여 사전 리스트 생성
p = Preprocess()
dict = []
for c in corpus_data:
  pos = p.pos(c[1])
  for k in pos:
    dict.append(k[0]) # ('헬로', 'NNP')

# print(dict[1])

# 사전에 사용될 워드 인덱스 생성
tokenizer = preprocessing.text.Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

# 사전 파일 생성
f = open(os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), 'wb')
try:
  pickle.dump(word_index, f)
except Exception as e:
  print('error: ', e)
finally:
  f.close()

print('사전 파일 생성 완료!!')
