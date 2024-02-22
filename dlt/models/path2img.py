import cv2
import os
import glob
from PIL import Image
import torch
from models.clip_encoder import CLIPModule
from PIL import Image

image_path = 'slide_image'  # 폴더 경로 설정
slide_images = {}  # 슬라이드 번호별 이미지 저장을 위한 딕셔너리

clipmodule = CLIPModule()
image_path = "slide_image/"






