
import cv2
from ultralytics import YOLO

# 모델 생성
model = YOLO("models/yolov8n.pt") # cpu에서 동영상 처리
# 카메라 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 한 프레임씩 읽기
    succ, frame = cap.read() # 한 프레임 읽기
    if not succ:
        print("연결 종료")
        break
    # BGR -> RGB, 좌우반전
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = cv2.flip(input_img, 1) # 1: 좌우반전 / 0: 상하반전
    ## 모델로 input_img를 입력해서 Detection 
    result = model(source= input_img, conf = 0.5, show = False)[0] # 리스트로 반환, 하나의 프레임[0] 가져오기
    
    ## Detection 결과를 원본 이미지에 출력  -> bbox, class 이름, class일 확률을 출력(그린다)
    cls_name= result.names
    
    boxes = result.boxes 
    for i in range(len(boxes)):  # 하나의 object
        # object의 좌표
        l_x = int(boxes.xyxy[i][0].item()) # 좌 x
        l_y = int(boxes.xyxy[i][1].item()) # 좌 y
        r_x = int(boxes.xyxy[i][2].item()) # 우 x
        r_y = int(boxes.xyxy[i][3].item()) # 우 y
        
        # object의 confidence score
        conf_sc = round(boxes.conf[i].item(),2)
        label = f"{cls_name[cls_idx]}-{conf_sc}"   # 라벨 변수 
        
        # 박스 그리기 bbox 
        cv2.rectangle(input_img,
                      (l_x,l_y), # 좌상단 좌표
                      (r_x,r_y), # 우상단 좌표 
                      (255,0,0), # 색상 지정 rgb
                      3 # 선두께
                     )
                      
        # 텍스트 그리기 - class이름, conf score
        cls_idx = boxes.cls[i].item()
        cv2.putText(input_img, label,(l_x,l_y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2, cv2.LINE_AA)
                
    cv2.imshow("frame", cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR)) 
    if cv2.waitKey(1) == 27: # esc 
        break

cap.release()
cv2.destroyAllWindows() 


# input_image 안씀
# boxes로 받고 인덱스만 매기기 -> box자체를 for문 돌리면 ?
