from PIL import Image
import os

def create_collage(batch, ids, geometries, canvas_size, base_path, scaling_size, mean_0):
    
    ppt_name = ids[0].split('/')[0]
    slide_name = ids[0].split('/')[1]
    slide_name = slide_name.split('_Shape')[0] + '_backgorund.png'

    
    background_path = os.path.join(base_path, ppt_name, slide_name)
    
    # 배경 이미지 로드
    background = Image.open(background_path)
    
    # 배경 이미지 크기가 캔버스 크기와 다른 경우 메시지 출력
    if background.size != canvas_size:
        print(f"Warning: The background image size {background.size} is different from the canvas size {canvas_size}.")
    
    # 배경 이미지를 캔버스 크기에 맞게 조정
    background = background.resize(canvas_size)

    # 배경 이미지가 RGBA 모드가 아닌 경우 변환
    if background.mode != 'RGBA':
        background = background.convert('RGBA')
        print('The background image is not RGBA')
    
    # 캔버스를 배경 이미지로 초기화
    collage = background

    for i, (file_name, geometry) in enumerate(zip(ids, geometries)):
        image_path = os.path.join(base_path, file_name)
        img = Image.open(image_path)
        
        x_scale = 1920.0 / scaling_size
        y_scale = 1080.0 / scaling_size
        w_scale = 1920.0 / scaling_size
        h_scale = 1080.0 / scaling_size

        # Geometry 정보 추출 및 처리
        
        x, y, w, h, r, z= geometry
        if mean_0 ==True:
            x = (x+ scaling_size)/2
            y = (y+ scaling_size)/2
            w = (w+ scaling_size)/2
            h = (h+ scaling_size)/2
            
        x = x * x_scale
        y = y * y_scale
        w = w * w_scale
        h = h * h_scale
        r = r * 360.0
        z = batch[i][5] * len(ids)
       

        if w < 1:
            w = 1
            print("############### width가 1 보다 작음 ##################")
        if h < 1:
            h = 1
            print("############### height가 1 보다 작음 ##################")
        
        img = img.resize((int(w), int(h))).rotate(r, expand=True)

        # 이미지가 RGBA 모드가 아닌 경우 변환
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # 투명도 마스크로 알파 채널 사용
        mask = img.split()[3]

        # 이미지 배치 위치 계산
        img_center_x, img_center_y = img.size[0] // 2, img.size[1] // 2
        top_left_x = int(x - img_center_x)
        top_left_y = int(y - img_center_y)

        # 이미지 캔버스에 배치
        collage.paste(img, (top_left_x, top_left_y), mask)

    return collage

############### z 고려하는 버전! => geometries줄 때 padding 부분 삭제하기 ###############

# from PIL import Image
# import os

# def create_collage(ids, geometries, canvas_size, base_path):
#     ppt_name = ids[0].split('/')[0]
#     slide_name = ids[0].split('/')[1]
#     slide_name = slide_name.split('_Shape')[0] + '_background.png'
    
#     background_path = os.path.join(base_path, ppt_name, slide_name)
#     background = Image.open(background_path)
    
#     if background.size != canvas_size:
#         print(f"Warning: The background image size {background.size} is different from the canvas size {canvas_size}.")
    
#     background = background.resize(canvas_size)
#     if background.mode != 'RGBA':
#         background = background.convert('RGBA')
#         print('The background image is not RGBA')
    
#     collage = background

#     # ids, geometries, image_paths를 z 값에 따라 정렬
#     items = zip(ids, geometries)
#     sorted_items = sorted(items, key=lambda x: x[1][-1])  # z 값(geometries의 마지막 요소)에 따라 정렬
    
#     for file_name, geometry in sorted_items:
#         image_path = os.path.join(base_path, file_name)
#         img = Image.open(image_path)
        
#         x, y, w, h, r, z = geometry
#         x = x * canvas_size[0]
#         y = y * canvas_size[1]
#         w = w * canvas_size[0]
#         h = h * canvas_size[1]
#         r = r * 360

#         if w < 1: w = 1
#         if h < 1: h = 1
        
#         img = img.resize((int(w), int(h))).rotate(r, expand=True)
#         if img.mode != 'RGBA':
#             img = img.convert('RGBA')
        
#         mask = img.split()[3]
#         img_center_x, img_center_y = img.size[0] // 2, img.size[1] // 2
#         top_left_x = int(x - img_center_x)
#         top_left_y = int(y - img_center_y)

#         collage.paste(img, (top_left_x, top_left_y), mask)

#     return collage
