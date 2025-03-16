#!/bin/bash

PORT=9000  # 재시작하려는 특정 포트 번호

# 지정한 포트에서 실행 중인 Uvicorn 프로세스의 PID를 가져와서 종료
for pid in $(ps aux | grep uvicorn | grep -- "--port $PORT" | grep -v grep | awk '{print $2}')
do
    echo "Killing process ID $pid on port $PORT"
    kill -9 $pid
done

echo "All Uvicorn processes on port $PORT have been terminated."

# 지정한 포트에서 Uvicorn 애플리케이션 재시작
nohup uvicorn main:app --host 0.0.0.0 --port $PORT &

# tail -f nohup.out
sleep 20

# 재시작 후 해당 포트에 대한 curl 테스트
curl http://localhost:$PORT/

