import cv2

def resize_and_show(img):
    # 이미지 데이터 타입 확인 및 변환
    if img.dtype != 'uint8':
        img = img.astype('uint8')

    # 이미지 크기 정보 로깅
    print(f"Image size: {img.shape}")

    # 이미지 표시
    return img


  
BG_COLOR = (255, 255, 255) # white
MASK_COLOR = (192, 192, 192) # gray
