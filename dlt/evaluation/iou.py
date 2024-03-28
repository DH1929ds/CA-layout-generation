import torch 
import torch.nn.functional as F 
from shapely.geometry import Polygon
import numpy as np
import math

def transform(real_geometry, pred_geometry, scaling_size, mask, mean_0):
    real_geometry = real_geometry[:,:,:4]
    pred_geometry = pred_geometry[:,:,:4]
    mask = mask[:,:,:4] 
    mask_cond = mask.sum(dim=-1) != 0
    selected_indices = mask_cond.unsqueeze(-1).expand_as(real_geometry)

    real_geometry = real_geometry[selected_indices].view(-1, 4)
    pred_geometry = pred_geometry[selected_indices].view(-1, 4)
    
    x_scale = 1920.0 / scaling_size
    y_scale = 1080.0 / scaling_size
    w_scale = 1920.0 / scaling_size
    h_scale = 1080.0 / scaling_size    
    # r_scale = 360.0 / scaling_size  
    
    if mean_0 == True:
        real_geometry = (real_geometry + scaling_size)/2 # 각자 normalize 한 것에 맞춰서 범위를 (0,1)로 변환할 것!
    real_geometry[:,0]*=x_scale
    real_geometry[:,1]*=y_scale
    real_geometry[:,2]*=w_scale
    real_geometry[:,3]*=h_scale
    # real_geometry[:,4]*=r_scale
    
    if mean_0 == True:
        pred_geometry = (pred_geometry + scaling_size)/2 # 각자 normalize 한 것에 맞춰서 범위를 (0,1)로 변환할 것!
    pred_geometry[:,0]*=x_scale
    pred_geometry[:,1]*=y_scale
    pred_geometry[:,2]*=w_scale
    pred_geometry[:,3]*=h_scale
    # pred_geometry[:,4]*=r_scale
    
    real_box = real_geometry[:, :4]  # [xi, yi, wi, hi]
    predicted_box = pred_geometry[:, :4]  # [xf, yf, wf, hf]

    return real_box, predicted_box 

# def calculate_iou(box1, box2):
#     # box1, box2: [x, y, w, h]
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2

#     # Calculate coordinates for each bounding box
#     x_min = max(x1 - w1/2, x2 - w2/2)
#     y_min = max(y1 - h1/2, y2 - h2/2)
#     x_max = min(x1 + w1/2, x2 + w2/2)
#     y_max = min(y1 + h1/2, y2 + h2/2)

#     # Calculating the area of the intersection
#     intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

#     # Calculate the area of each bounding box
#     area_box1 = w1 * h1
#     area_box2 = w2 * h2

#     # Calculating the area of the union area
#     union_area = area_box1 + area_box2 - intersection_area

#     # Calculating IoU
#     ious = intersection_area / union_area

#     return ious

# def get_iou(real_box, predicted_box):
#     # IoU 계산
#     iou = torch.zeros(real_box.shape[0])
    
#     for i in range(real_box.shape[0]):
#         iou[i] = calculate_iou_rotated(real_box[i], predicted_box[i])
    
#     return iou


### rotation 고려 안 했을 때의 iou (병렬로 계산)
def get_iou(true_boxes, pred_boxes):
    # Extract coordinates from boxes
    x1, y1, w1, h1 = true_boxes[:, :, 0], true_boxes[:, :, 1], true_boxes[:, :, 2], true_boxes[:, :, 3]
    x2, y2, w2, h2 = pred_boxes[:, :, 0], pred_boxes[:, :, 1], pred_boxes[:, :, 2], pred_boxes[:, :, 3]

    # Calculate coordinates for each bounding box
    x_min = torch.maximum(x1 - w1/2, x2 - w2/2)
    y_min = torch.maximum(y1 - h1/2, y2 - h2/2)
    x_max = torch.minimum(x1 + w1/2, x2 + w2/2)
    y_max = torch.minimum(y1 + h1/2, y2 + h2/2)

    # Calculating the area of the intersection
    intersection_area = torch.maximum(torch.zeros_like(x_min), x_max - x_min) * torch.maximum(torch.zeros_like(y_min), y_max - y_min)

    # Calculate the area of each bounding box
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Calculating the area of the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculating IoU
    ious = intersection_area / torch.maximum(union_area, torch.tensor(1e-8))

    return ious

def print_results(true_boxes, pred_boxes):
    # IoU 계산
    ious = get_iou(true_boxes, pred_boxes)

    print(f"IoUs: {ious}")
    print(f"Mean IoU: {get_mean_iou(true_boxes, pred_boxes)}")

def get_mean_iou(true_boxes, pred_boxes, mask):
    ious = get_iou(true_boxes, pred_boxes)

    ele_num = mask.sum(dim=2) !=0
    ele_num = ele_num.sum(dim=1)
    mean_ious = ious.sum(dim=1)/ele_num

    return mean_ious


###################### Obtaining iou of the two rotated bounding boxes  ####################
def get_mean_iou_rotated(true_boxes, pred_boxes, mask):
    # true_boxes의 장치를 가져옴
    device = true_boxes.device

    # ele_num 계산
    ele_num = mask.sum(dim=2) != 0
    ele_num = ele_num.sum(dim=1).to(device)  # 같은 장치로 이동

    # ious 초기화: 각 요소에 대해 0으로 채워진 텐서 생성, 같은 장치에 생성
    ious = torch.zeros(true_boxes.shape[0], true_boxes.shape[1], device=device)

    for i in range(true_boxes.shape[0]):
        for j in range(true_boxes.shape[1]):
            box1 = true_boxes[i, j, :]
            box2 = pred_boxes[i, j, :]
            ious[i, j] = calculate_iou_rotated(box1, box2)  # 해당 함수도 장치에 맞게 구현되어야 함

    # 각 배치에 대해 ious를 합산하고, 유효한 요소 수로 나누어 평균 IoU 계산
    sum_ious = ious.sum(dim=1)
    mean_ious = sum_ious / ele_num.float()  # 연산을 위해 float 타입으로 변환할 수 있음

    return mean_ious


def calculate_iou_rotated(box1, box2):
    """
    box1, box2: [cx, cy, width, height, angle]
    """
    cx1, cy1, w1, h1, angle1 = box1[0], box1[1], box1[2], box1[3], box1[4]
    cx2, cy2, w2, h2, angle2 = box2[0], box2[1], box2[2], box2[3], box2[4]
    
    # Calculate the rotated coordinates of the corners
    corners1 = torch.tensor([[-w1 / 2, -h1 / 2],
                             [w1 / 2, -h1 / 2],
                             [w1 / 2, h1 / 2],
                             [-w1 / 2, h1 / 2]])

    corners2 = torch.tensor([[-w2 / 2, -h2 / 2],
                             [w2 / 2, -h2 / 2],
                             [w2 / 2, h2 / 2],
                             [-w2 / 2, h2 / 2]])

    rotation_matrix1 = torch.tensor([[torch.cos(angle1), -torch.sin(angle1)],
                                     [torch.sin(angle1), torch.cos(angle1)]])

    rotation_matrix2 = torch.tensor([[torch.cos(angle2), -torch.sin(angle2)],
                                     [torch.sin(angle2), torch.cos(angle2)]])

    rotated_corners1 = torch.mm(corners1, rotation_matrix1.t()) + torch.tensor([cx1, cy1])
    rotated_corners2 = torch.mm(corners2, rotation_matrix2.t()) + torch.tensor([cx2, cy2])

    # Calculate the intersection area
    intersection_area = polygon_intersection_area(rotated_corners1, rotated_corners2)

    # Calculate the union area
    union_area = polygon_area(rotated_corners1) + polygon_area(rotated_corners2) - intersection_area

    # Calculate IoU
    iou = intersection_area / max(union_area, 1e-8)

    return iou  # Convert to Python float 

def polygon_intersection_area(corners1, corners2):
    """
    Calculate the area of intersection between two polygons.
    """
    # Implementation of the Sutherland-Hodgman algorithm to clip polygons
    corners1_np = corners1.numpy()
    corners2_np = corners2.numpy()
    
    poly1 = Polygon(corners1_np)
    poly2 = Polygon(corners2_np)
    
    intersection_area = poly1.intersection(poly2).area
    return intersection_area

def polygon_area(corners):
    """
    Convert torch tensor corners to a numpy array and calculate the polygon area.
    """
    corners_np = corners.numpy()
    poly = Polygon(corners_np)
    return poly.area

def calculate_iou(box1, box2):
    # box1, box2: [x, y, w, h]
    x1, y1, w1, h1, r, z = box1
    x2, y2, w2, h2, r, z = box2

    # Calculate coordinates for each bounding box
    x_min = max(x1 - w1/2, x2 - w2/2)
    y_min = max(y1 - h1/2, y2 - h2/2)
    x_max = min(x1 + w1/2, x2 + w2/2)
    y_max = min(y1 + h1/2, y2 + h2/2)

    # Calculating the area of the intersection
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the area of each bounding box
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Calculating the area of the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculating IoU
    ious = intersection_area / union_area

    return ious

def get_iou_slide(real_box, predicted_box):
    # 0인 열 제거 전에 real_box와 predicted_box에서 모든 값이 0인 열을 찾아 제거합니다.
    non_zero_indices = torch.logical_and(real_box.sum(dim=1) != 0, predicted_box.sum(dim=1) != 0)
    filtered_real_box = real_box[non_zero_indices]
    filtered_predicted_box = predicted_box[non_zero_indices]
    
    # IoU 계산
    num_boxes = filtered_real_box.shape[0]
    iou = torch.zeros(num_boxes)
    
    for i in range(num_boxes):
        iou[i] = calculate_iou(filtered_real_box[i], filtered_predicted_box[i])
    
    # 0이 아닌 IoU 값이 하나라도 있는 경우 평균 IoU 값을 계산합니다.
    # 모든 IoU 값이 0인 경우, avg_iou 값을 0으로 설정합니다.
    if len(iou) > 0:
        avg_iou = torch.mean(iou)
    else:
        avg_iou = torch.tensor(0.0)
    
    return avg_iou