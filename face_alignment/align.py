import sys
import os

from AdaFace.face_alignment import mtcnn
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
import logging

logger = logging.getLogger("app.py")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MTCNN_MODEL = mtcnn.MTCNN(device=DEVICE, crop_size=(112, 112))
logger.info(f"Loaded MTCNN model for face alignment with device: {DEVICE}")

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None, max_num_faces=1):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = MTCNN_MODEL.align_multi(img, limit=max_num_faces)
        faces = faces[:max_num_faces]
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        faces = None

    return faces


