import os
import sys
from pathlib import Path

import pickle
import pandas as pd
import tensorflow as tf

from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, concatenate

if 'JAVA_HOME' not in os.environ:
  os.environ["JAVA_HOME"] = JAVA_HOME

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent 
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from config.GlobalParams import * 

# 데이터 호출
train_file = os.path.join(root_dir, 'model', 'intent', 'total_train_data.csv')
data = pd.read_csv(train_file)

# print(data[:5])

# 질문과 의도 문자열을 리스트로 변환 저장
queries = data['query'].tolist()
intents = data['intent'].tolist()

# 단어 인덱스와 사용자 정의 사전 파일 호출 및 저장
from utils.preprocess import Preprocess
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# 단어 시퀀스 생성
sequences = []
for sentence in queries:
  pos = p.pos(sentence)
  keywords = p.get_keywords(pos, without_tag=True)
  seq = p.get_wordidx_sentence(keywords)
  sequences.append(seq)

# print(sequences[:5])

# 단어 인텍스 시퀀스 벡터 생성 및 입력 크기 지정(maxlen)
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')

# 학습용, 검증용, 테스트용 데이터셋 분리: 7:2:1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

# print(len(ds))

# 데이터 분리
train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

# print(len(train_ds), len(val_ds), len(test_ds))

# 하이퍼 파라미터
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1

# CNN 모델 정의
input_layer = Input(shape=(15,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=15)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# 합성곱층 추가 및 통합 
conv1 = Conv1D()(dropout_emb)
pool1 = GlobalMaxPooling1D()(conv1)

conv2 = Conv1D()(dropout_emb)
pool2 = GlobalMaxPooling1D()(conv2)

conv3 = Conv1D()(dropout_emb)
pool3 = GlobalMaxPooling1D()(conv3)

# 통합
concat = concatenate([pool1, pool2, pool3])

# 완전 연결층