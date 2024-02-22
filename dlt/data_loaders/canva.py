import random
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from functools import partial

class CanvaLayout(Dataset):
    def __init__(self, json_path, clip_json_path, max_num_com: int = 20, scaling_size=5, z_scaling_size=0.01, mean_0 = True):
        #self.presentation_size = presentation_size
        self.max_num_element = max_num_com
        self.scaling_size = scaling_size 
        self.z_scaling_size = z_scaling_size
        self.mean_0 = mean_0
        self.data = self.process(json_path,clip_json_path)
    
    def normalize_geometry(self, slide, content, num_elements):
        # Normalize left, top, width, height

        left = content['left']
        top = content['top']
        width = content['width']
        height = content['height']
        
        xc = left + width / 2.
        yc = top + height / 2.
        
        x_scale = 1920.0 / self.scaling_size
        y_scale = 1080.0 / self.scaling_size
        w_scale = 1920.0 / self.scaling_size
        h_scale = 1080.0 / self.scaling_size
        z_scale = 20.0/ self.z_scaling_size # max 요소 20개
        
        if self.mean_0 == True:
            x = xc / x_scale *2 -self.scaling_size
            y = yc / y_scale *2 -self.scaling_size
            w = content['img_width'] / w_scale *2 - self.scaling_size
            h = content['img_height'] / h_scale*2 - self.scaling_size
            w = content['img_width'] / w_scale*2 - self.scaling_size
            h = content['img_height'] / h_scale*2 - self.scaling_size
        else:
            x = xc / x_scale
            y = yc / y_scale            
            w = content['img_width'] / w_scale
            h = content['img_height'] / h_scale
        # Normalize rotation (optional)
        r = content['rotation'] / 360.0  # Assuming rotation is in degrees

        # Normalize z_index for each slide separately
        z = content['z_index'] / z_scale # max num comp로 하든말든
        return [x, y, w, h, r, z]

    def process(self, json_path, clip_json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        with open(clip_json_path, 'r') as f:
            clip_data = json.load(f)

        processed_data = {"geometry": [], "image_features": [], "ids": [], "type": []} 
        for presentation in data['presentations']:
            for slide in presentation['slides']:
                
                slide_geometry = []
                slide_image_features = []
                slide_ids = []  # 슬라이드별 id 저장을 위한 리스트
                slide_type = []
                num_elements = len(slide['contents'])  # Number of elements in the current slide
                for content in slide['contents']:
        
                    geometry = self.normalize_geometry(slide, content, num_elements)
                    slide_geometry.append(geometry)
                    
                    # Fetching image features from clip_data using ppt_name and image_file_name
                    image_file_name = content.get('image_file_name', '')
                    ppt_name = presentation['ppt_name']
                    image_features = clip_data.get(ppt_name, {}).get(image_file_name, [])
                    slide_image_features.append(image_features)
                    
                    content_id = f"{ppt_name}/{image_file_name}" 
                    slide_ids.append(content_id)
                    content_type = content.get('type', '')
                    slide_type.append(content_type)
                    
                processed_data["geometry"].append(slide_geometry)
                processed_data["image_features"].append(slide_image_features)
                processed_data["ids"].append(slide_ids)  # 슬라이드별 id 정보 추가
                processed_data["type"].append(slide_type)
        return processed_data

    def get_data(self):
        return self.data

    def mask_instance(self, geometry):
        # 나중에 
        return np.ones(6)  # Placeholder for the masked geometry

    def pad_instance(self, geometry):
        padded_geometry = np.pad(geometry, pad_width=((0,self. max_num_element - np.array(geometry).shape[0]), (0, 0)), constant_values=0.)
        return padded_geometry
    
    def pad_instance_type(self, cat):
        num_pad_elements = max(0, self.max_num_element - len(cat))
        # 1차원 배열에 대한 패딩, 배열의 끝에만 패딩을 추가
        padded_cat = np.pad(cat, pad_width=(0, num_pad_elements), constant_values=0)
        return padded_cat

    def process_data(self, idx):
        geometry = self.data['geometry'][idx]
        cat = self.data['type'][idx]
        padding_mask = np.ones(np.array(geometry).shape)
        mask = self.mask_instance(geometry)
        geometry = self.pad_instance(geometry)
        cat = self.pad_instance_type(cat).reshape((-1,1))
        padding_mask = self.pad_instance(padding_mask)
        image_features = self.data['image_features'][idx]
        image_features = np.squeeze(image_features, axis=1)
        image_features = self.pad_instance(image_features)
        
        ids = self.data['ids'][idx]  # id 정보 로드
        
        cat[cat=='freeform']=1
        cat[cat=='picture']=2
        cat[cat=='auto_shape']=3
        cat[cat=='text_box']=4
        cat[cat=='group']=5
        cat[cat=='table']=6
        cat[cat=='media']=6
        cat[cat=='0'] = 0
       
        cat = cat.reshape((-1,))
        
        return {
            "geometry": np.array(geometry).astype(np.float32),
            "image_features": np.array(image_features).astype(np.float32),
            "padding_mask": padding_mask.astype(np.int32),
            "ids": ids,  # id 정보 반환
            "cat": cat.astype(int)
        }

    def __getitem__(self, idx):
        sample = self.process_data(idx)
        return sample

    def __len__(self):
        return len(self.data['geometry'])


##########################type X #####################
# import random
# import json
# import numpy as np
# from pathlib import Path
# from torch.utils.data import Dataset
# from functools import partial

# class CanvaLayout(Dataset):
#     def __init__(self, json_path, clip_json_path, max_num_com: int = 20, scaling_size=5, z_scaling_size=0.01, mean_0 = True):
#         #self.presentation_size = presentation_size
#         self.max_num_element = max_num_com
#         self.scaling_size = scaling_size 
#         self.z_scaling_size = z_scaling_size
#         self.data = self.process(json_path,clip_json_path)
#         self.mean_0 = mean_0
    
#     def normalize_geometry(self, slide, content, num_elements):
#         # Normalize left, top, width, height

#         left = content['left']
#         top = content['top']
#         width = content['width']
#         height = content['height']
        
#         xc = left + width / 2.
#         yc = top + height / 2.
        
#         x_scale = 1920.0 / self.scaling_size
#         y_scale = 1080.0 / self.scaling_size
#         w_scale = 1920.0 / self.scaling_size
#         h_scale = 1080.0 / self.scaling_size
#         z_scale = 20.0/ self.z_scaling_size # max 요소 20개
        
#         if self.mean_0 == True:
#             x = xc / x_scale
#             y = yc / y_scale
#             if 'img_width' not in content:
            
#                 w = content['img_width'] / w_scale *2 - self.scaling_size
#                 h = content['img_height'] / h_scale*2 - self.scaling_size
#             else:
#                 w = content['img_width'] / w_scale*2 - self.scaling_size
#                 h = content['img_height'] / h_scale*2 - self.scaling_size
#         else:
#             x = xc / x_scale
#             y = yc / y_scale
#             if 'img_width' not in content:
            
#                 w = content['img_width'] / w_scale
#                 h = content['img_height'] / h_scale
#             else:
#                 w = content['img_width'] / w_scale
#                 h = content['img_height'] / h_scale
#         # Normalize rotation (optional)
#         r = content['rotation'] / 360.0  # Assuming rotation is in degrees

#         # Normalize z_index for each slide separately
#         z = content['z_index'] / z_scale
#         return [x, y, w, h, r, z]

#     def process(self, json_path, clip_json_path):
#         with open(json_path, 'r') as f:
#             data = json.load(f)
        
#         with open(clip_json_path, 'r') as f:
#             clip_data = json.load(f)

#         processed_data = {"geometry": [], "image_features": [], "ids": []} 
#         for presentation in data['presentations']:
#             for slide in presentation['slides']:
                
#                 slide_geometry = []
#                 slide_image_features = []
#                 slide_ids = []  # 슬라이드별 id 저장을 위한 리스트
#                 num_elements = len(slide['contents'])  # Number of elements in the current slide
#                 for content in slide['contents']:
        
#                     geometry = self.normalize_geometry(slide, content, num_elements)
#                     slide_geometry.append(geometry)
                    
#                     # Fetching image features from clip_data using ppt_name and image_file_name
#                     image_file_name = content.get('image_file_name', '')
#                     ppt_name = presentation['ppt_name']
#                     image_features = clip_data.get(ppt_name, {}).get(image_file_name, [])
#                     slide_image_features.append(image_features)
                    
#                     content_id = f"{ppt_name}/{image_file_name}" 
#                     slide_ids.append(content_id)

#                 processed_data["geometry"].append(slide_geometry)
#                 processed_data["image_features"].append(slide_image_features)
#                 processed_data["ids"].append(slide_ids)  # 슬라이드별 id 정보 추가
#         return processed_data

#     def get_data(self):
#         return self.data

#     def mask_instance(self, geometry):
#         # 나중에 
#         return np.ones(6)  # Placeholder for the masked geometry

#     def pad_instance(self, geometry):
#         num_pad_elements = max(0, self.max_num_element - len(geometry))
#         #padded_geometry = np.pad(geometry, pad_width=((0, num_pad_elements), (0, 0)), constant_values=0.0)
#         padded_geometry = np.pad(geometry, pad_width=((0,self. max_num_element - np.array(geometry).shape[0]), (0, 0)), constant_values=0.)
#         return padded_geometry

#     def process_data(self, idx):
#         geometry = self.data['geometry'][idx]
#         padding_mask = np.ones(np.array(geometry).shape)
#         mask = self.mask_instance(geometry)
#         geometry = self.pad_instance(geometry)
#         padding_mask = self.pad_instance(padding_mask)
        
#         image_features = self.data['image_features'][idx]
#         image_features = np.squeeze(image_features, axis=1)
#         image_features = self.pad_instance(image_features)
        
#         ids = self.data['ids'][idx]  # id 정보 로드
        
#         return {
#             "geometry": np.array(geometry).astype(np.float32),
#             "image_features": np.array(image_features).astype(np.float32),
#             "padding_mask": padding_mask.astype(np.int32),
#             "ids": ids  # id 정보 반환
#         }

#     def __getitem__(self, idx):
#         sample = self.process_data(idx)
#         return sample

#     def __len__(self):
#         return len(self.data['geometry'])


