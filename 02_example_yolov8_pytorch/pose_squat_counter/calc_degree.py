
import math
import numpy as np

def calc_degree(p1, p2, p3):
    ### p1 : 어깨 좌표, p2 :팔꿈치 좌표, p3: 손목 좌표
    p1_2 = p1 - p2 #원점과 계산
    p3_2 = p3 - p2

    if p1_2[0] == 0 or p3_2[0] == 0:
        return None
    # a : p1과 X축간의 각도
    # b : p3와 X축간의 각도
    # p2 : 원점
    else:
        a = abs(math.atan(p1_2[1]/p1_2[0])) * 180 / math.pi    # 각도 구하는 코드 
        b = abs(math.atan(p3_2[1]/p3_2[0])) * 180 / math.pi
    
        sign1 = np.sign(p1 - p2)   # p2를 원점으로 해서 p1이 몇사분면에 있는지 (+,+) 
        sign2 = np.sign(p3 - p2)   
        x_sign, y_sign = sign1 * sign2
    
        result = None
        if x_sign == 1 and y_sign==1: # a - b
            result = a - b  
    
        elif x_sign == -1 and y_sign == 1: # 180 - (a + b)
            result = 180- (a + b)
    
        elif x_sign == 1 and y_sign == -1: # a + b
            result = a + b
    
        elif x_sign == -1 and y_sign == -1:
            print(p1, p3)
            if p1[1] < p3[1]:
                result = (180 -a) + b
            else:
                result = (180 - b) + a

    return result
