# squat counter  
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from calc_degree import calc_degree

model = YOLO("models/yolov8n-pose.pt")

path = r"practice/squat_count.mp4"
vidcap = cv2.VideoCapture(path)

fps = vidcap.get(cv2.CAP_PROP_FPS)   # 프레임 속도(1초당 프레임수)
delay = int(1000/fps*2) # waitkey는 밀리초를 인자로 받으므로 역수에 1000곱하기 

# 스쿼트 판별 변수
prev_squat_flag=False
cnt = 0

while vidcap.isOpened():
    # 한 프레임씩 읽기
    succ, frame = vidcap.read()
    if not succ:
        print("연결 종료")
        break

    # BGR->RGB 변환 해 모델 추론
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(input_img, show=False, verbose = False, save= False)[0] 
    keypoints= result.keypoints

    ## segmentation 결과를 원본 이미지에 출력  -> 다리 선, 각도 85도 이하면 횟수 카운트  
    for keypoint in keypoints: # 한 object
        l_hip = keypoints.xy[0,11].type(torch.int32).numpy()
        l_knee = keypoints.xy[0,13].type(torch.int32).numpy()
        l_ank = keypoints.xy[0,15].type(torch.int32).numpy()
        
        degree = calc_degree(l_hip, l_knee, l_ank)
        # print(degree, type(degree))

        # 스쿼트 판별
        squat_flag = False   # frame마다 False로 초기화 # 85도 이하인지 판별
        if degree is not None: 
            degree=int(degree)
            if degree <= 85:
                squat_flag=  True
            if squat_flag and not prev_squat_flag:  # prev_squat_flag는 frame마다 초기화x  # 이전에도 85이하인지 판별 
                # print(degree, type(degree),"1회")
                cnt+=1
            prev_squat_flag= squat_flag
        
            # 원본 이미지에 그리기
            pts = np.array([l_hip, l_knee, l_ank])
            cv2.polylines(input_img, [pts],False, (255,0,0),1, lineType = cv2.LINE_AA,)
            cv2.putText(input_img, f"{int(degree)}", l_hip+50, cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2, cv2.LINE_AA)
            cv2.putText(input_img, f"{cnt}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3,lineType=cv2.LINE_AA)

    cv2.imshow("frame", cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR))
    if cv2.waitKey(delay-50) == 27:
        break

print(f"스쿼트를 {cnt}회 했습니다.")
vidcap.release()
cv2.destroyAllWindows()
