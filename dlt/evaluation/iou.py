## made by 형준
import torch 
import torch.nn.functional as F 

def transform(real_geometry, pred_geometry, scaling_size,mask, mean_0):
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
    x1, y1, w1, h1 = true_boxes[:, 0], true_boxes[:, 1], true_boxes[:, 2], true_boxes[:, 3]
    x2, y2, w2, h2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

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
    ious = intersection_area / torch.maximum(union_area, torch.tensor(1e-6))

    return ious

def print_results(true_boxes, pred_boxes):
    # IoU 계산
    ious = get_iou(true_boxes, pred_boxes)

    print(f"IoUs: {ious}")
    print(f"Mean IoU: {get_mean_iou(true_boxes, pred_boxes)}")

def get_mean_iou(true_boxes, pred_boxes):
    ious = get_iou(true_boxes, pred_boxes)
    mean_iou = torch.mean(ious).item()
    return mean_iou


###################### Obtaining iou of the two rotated bounding boxes  ####################

def calculate_iou_rotated(box1, box2):
    """
    box1, box2: [cx, cy, width, height, angle]
    """
    cx1, cy1, w1, h1, angle1 = box1
    cx2, cy2, w2, h2, angle2 = box2
    
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
    union_area = polygon_area(corners1) + polygon_area(corners2) - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou.item()  # Convert to Python float 

def polygon_intersection_area(poly1, poly2):
    """
    Calculate the area of intersection between two polygons.
    """
    # Implementation of the Sutherland-Hodgman algorithm to clip polygons
    intersection_points = []

    for i in range(len(poly1)):
        next_index = (i + 1) % len(poly1)
        clip_line = poly1[i], poly1[next_index]
        intersection_points.extend(clip_line_by_polygon(clip_line, poly2))

    # Calculate the area of the clipped polygon
    area = polygon_area(intersection_points)
    return area

def clip_line_by_polygon(clip_line, polygon):
    """
    Clip a line segment against a convex polygon.
    """
    result = []
    start, end = clip_line

    for i in range(len(polygon)):
        next_index = (i + 1) % len(polygon)
        edge = polygon[i], polygon[next_index]

        if is_inside(start, edge):
            if is_inside(end, edge):
                result.append(end)
            else:
                intersection_point = line_intersection(start, end, edge[0], edge[1])
                result.append(intersection_point)
        elif is_inside(end, edge):
            intersection_point = line_intersection(start, end, edge[0], edge[1])
            result.append(intersection_point)

    return result

def is_inside(point, edge):
    """
    Check if a point is inside or outside of an edge.
    """
    return (edge[1][0] - edge[0][0]) * (point[1] - edge[0][1]) > (edge[1][1] - edge[0][1]) * (point[0] - edge[0][0])

def line_intersection(p1, q1, p2, q2):
    """
    Calculate the intersection point of two lines.
    """
    x1, y1 = p1
    x2, y2 = q1
    x3, y3 = p2
    x4, y4 = q2
    
    determinant = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if determinant == 0:
        return torch.tensor([0.0, 0.0])  # Lines are parallel
    
    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / determinant
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / determinant
    
    return torch.tensor([intersection_x, intersection_y])

def polygon_area(points):
    """
    Calculate the area of a polygon using the shoelace formula. (신발끈 공식으로 삼각형 넓이 구하기)
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = torch.abs(area) / 2.0
    return area

