import os
import sys
from pathlib import Path # 경로 처러 모듈

if 'JAVA_HOME' not in os.environ:
  os.environ["JAVA_HOME"] = JAVA_HOME

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent # 현재 경로
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from config.GlobalParams import * # Global Params 내부에 지정된 값 가져온다

# 사용자 사전 파일 절대 경로 생성
user_dic_path = os.path.join(root_dir, 'test', 'user_dic.tsv')

# 테스트 문장
sent = '내일 오전 10시에 탕수육 주문하고 싶어요.'

try:
  from utils.preprocess import Preprocess

  # 전처리 객체 생성
  p = Preprocess(userdic=user_dic_path)

  # 형태소 분석기 실행
  pos = p.pos(sent)

  # 품사 태그와 단어를 함께 출력
  ret = p.get_keywords(pos, without_tag=False)
  print('태그를 포함한 형태소 분석: ', ret)

  ret = p.get_keywords(pos, without_tag=True)
  print('태그를 제거한 형태소 분석: ', ret)
except Exception as e:
  print('error: ', e)