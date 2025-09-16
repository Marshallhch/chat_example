import socket, json

# 엔진 서버 접속 정보
host = '127.0.0.1' # 디폴트 아이피
port = 5050

# 클라이언트 프로그램 시작
while True:
  print('질문: ')
  query = input().strip() # 사용자 질문 입력

  if(query == 'exit'):
    exit(0) # 프로그램 종료

  # 입력값이 비어있는지 확인
  if not query:
    print('질문을 입력해 주세요.')
    continue

  # 입력 확인
  print('입력 질문: ', query)
  print('입력 질문 타입: ', type(query))

  # 엔진 서버 연결
  mySocket = socket.socket()
  mySocket.connect((host, port))

  # 엔진 서버 질의 내용 전송
  query = str(query).strip()

  if not query:
    print('질문을 입력해 주세요.')
    continue

  json_data = {
    "Query": query,
    "BotType": "MyService"
  }

  # 전송 메시지 json 형식으로 변환
  message = json.dumps(json_data, ensure_ascii=False)
  print('전송 데이터: ', message)
  mySocket.send(message.encode())

  # 수신 데이터 저장
  data = mySocket.recv(2048).decode()

  # 수신된 데이터 확인 및 예외처리
  try:
    print('서버로 부터 받은 데이터: ', data)
    
    if not data:
      print('서버로부터 데이터를 받지 못했습니다.')
    else:
      ret_data = json.loads(data)
      print('봇 엔진 답변: ', ret_data['Answer'])
  except json.JSONDecodeError as e:
    print('json decode error: ', str(e))
  except Exception as e:
    print('error: ', str(e))

  # 엔진 서버 소켓 종료
  mySocket.close()