import os
from collections import defaultdict
import json
from typing import Any, List, Dict, Union, Tuple
import multiprocessing

from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import cv2
import numpy as np

from .interpreter import Box
from .FPVG import FPVG



class IMAGE_CLASS:
    def __init__(self,image_dict: Dict[str,Dict[str,Dict[str,Image.Image]]]) -> None:
        self.image_dict=image_dict
        self.loaded_count=0

    def get_image_dict(self, file_name: str) -> Dict[str, Dict[str,Image.Image]]:
        if self.loaded_count==0:
            self.load_img_batch()
        i=self.image_dict[file_name]
        del self.image_dict[file_name]
        self.loaded_count-=1
        # if self.image_dict[file_name][method_name] is {}:
        #     del self.image_dict[file_name][method_name]
        # if self.image_dict[file_name] is {}:
        #     del self.image_dict[file_name]
        return i

    def load_img_batch(self, batch_size: int = 50) -> None:
        i=0
        for file in self.image_dict.values():
            for method in file.values():
                for box in method.values():
                    box.load()
            i+=1
            if i == batch_size:
                self.loaded_count+=batch_size
                return
        self.loaded_count+=i
        return
        

class IMAGE_PREPROCESS:
    def __init__(self, image_path: str = None, input_file: str = None, detector_file_path: str = None) -> None:
        if image_path is None:
            raise ValueError("empty image path")
        if os.path.exists(image_path) is None:
            raise ValueError("wrong path")
        self.image_path=image_path
        with open(input_file) as f:
            lines = f.readlines()
            self.data = [json.loads(line) for line in lines]
        if detector_file_path:
            detector_file = open(detector_file_path)
            detections_list = json.load(detector_file)
            if isinstance(detections_list, dict):
                self.detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
            else:
                self.detections_map = defaultdict(list)
                for detection in detections_list:
                    self.detections_map[detection["image_id"]].append(detection["box"])

    def dynamic_color_width(self, box: Box, image: Image, max_width = 7, min_width = 3) -> (tuple, int):
        # 动态改变圈的宽度和颜色
        h, w = image.size
        width = int(((box[2] - box[0]) * (box[3] - box[1]) / (h * w)) * (max_width - min_width) + min_width)
        color = (255,0,0) # 红色

        cropped = image.crop(box)
        # 统计cropped的颜色，如果非常接近红色，color就为绿色，否则是红色
        cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        # 转换图像为HSV颜色空间
        hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        # 定义红色的HSV范围
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        # 创建一个掩码，标记图像中的红色区域
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        # 计算红色像素的百分比
        red_pixel_percentage = (np.count_nonzero(mask) / mask.size) * 10
        # 设置一个阈值，用于确定图像是否偏向于红色调
        threshold = 5  # 你可以根据需要调整阈值（大点）
        if red_pixel_percentage >= threshold:
            color = (128,0,128) # 紫色
        
        return color, width
        
    def resize2square(self, img: Image.Image) -> Image.Image:
        GRAY=(69,69,69)
        size=max(img.size)
        new_img=Image.new('RGB', (size,size), GRAY)
        new_img.paste(img)
        return new_img
    
    def crop_method(self, img, box, **kwargs):
        image_i = img.copy()
        return image_i.crop(box)

    #不改变原图片
    def circle_method(self, img, box, **kwargs):
        CIRCLE_COLOR = (240, 0, 30)  # 圈颜色
        CIRCLE_WIDTH = 3  # 圈宽度（大于0）
        image_i = img.copy().convert('RGBA')
        draw = ImageDraw.Draw(image_i)  # 用来画画的对象，直接改变图片

        if self.is_dynamic:
            CIRCLE_COLOR, CIRCLE_WIDTH = self.dynamic_color_width(box=box, image=image_i)
        draw.ellipse(
            (tuple(box[:2]), tuple(box[2:])),
            outline=CIRCLE_COLOR + (0,),
            width=CIRCLE_WIDTH
        )  # 直接用的 box 范围 后续改进
        return image_i.convert('RGB')

    def blur_method(self, img, box, image_name=None, box_index=None, **kwargs):
        image_i = img.copy()
        
        h, w =img.size
        enlarge = self.enlarge_boxes * (box[2] - box[0]) * (box[3] - box[1]) / (h * w)
        sam_box = [  # box 范围？ 后续改进调整范围
            max(box[0] - enlarge, 0),
            max(box[1] - enlarge, 0),
            min(box[2] + enlarge, image_i.width),
            min(box[3] + enlarge, image_i.height)
        ]
        # blur
        if self.fine_grained:
            mask = self.fpvg.get_mask(pic=image_i, image_name=image_name, box_index=box_index, box=sam_box, iter=self.sam_iter, method=self.sam_method)
        else:
            mask = Image.new('L', image_i.size, 0)
            # 在遮罩图像上绘制一个白色的圆，使圆内部变为不透明
            draw = ImageDraw.Draw(mask)
            draw.ellipse(box, fill=255)
        # 分割图像为两部分：圆内部和圆外部
        inner_circle = image_i.copy()
        outer_circle = image_i.copy()
        # 对圆外部的部分进行高斯模糊处理
        outer_circle = outer_circle.filter(ImageFilter.GaussianBlur(radius=100))  # 调整radius以控制模糊程度
        # 合并两部分图像，只对圆外部进行模糊
        result_image = Image.composite(inner_circle, outer_circle, mask)

        return result_image.convert('RGB')
    
    def gray_method(self, img, box, image_name=None, box_index=None, **kwargs):
        image_i = img.copy()
        
        h, w =img.size
        enlarge = self.enlarge_boxes * (box[2] - box[0]) * (box[3] - box[1]) / (h * w)
        sam_box = [  # box 范围？ 后续改进调整范围
            max(box[0] - enlarge, 0),
            max(box[1] - enlarge, 0),
            min(box[2] + enlarge, image_i.width),
            min(box[3] + enlarge, image_i.height)
        ]

        if self.fine_grained:
            mask = self.fpvg.get_mask(pic=image_i, image_name=image_name, box_index=box_index, box=sam_box, iter=self.sam_iter, method=self.sam_method)
        else:
            mask = Image.new('L', image_i.size, 0)
            # 在遮罩图像上绘制一个白色的圆，使圆内部变为不透明
            draw = ImageDraw.Draw(mask)
            draw.ellipse(box, fill=255)
        # 将原始图像转换为灰度图像
        grayscale_image = image_i.copy().convert('L').convert('RGB')
        # 创建一个新图像，将灰度图像与遮罩图像合并，只对圆外部进行灰度处理
        result_image = Image.composite(image_i, grayscale_image, mask)

        return result_image.convert('RGB')
    
    def edge_method(self, img, box, image_name=None, box_index=None, **kwargs):
        image_i = img.copy()
        
        EDGE_COLOR = (240, 0, 30)  # 圈颜色
        EDGE_WIDTH = 3  # 圈宽度（大于0）
        if self.is_dynamic:
            EDGE_COLOR, EDGE_WIDTH = self.dynamic_color_width(box=box, image=image_i)

        h, w =image_i.size
        enlarge = self.enlarge_boxes * (box[2] - box[0]) * (box[3] - box[1]) / (h * w)
        sam_box = [  # box 范围？ 后续改进调整范围
            max(box[0] - enlarge, 0),
            max(box[1] - enlarge, 0),
            min(box[2] + enlarge, image_i.width),
            min(box[3] + enlarge, image_i.height)
        ]
        image_array = np.array(image_i)
        mask = self.fpvg.get_mask(pic=image_i, image_name=image_name, box_index=box_index, box=sam_box, iter=self.sam_iter, method=self.sam_method)
        
        mask_array=np.uint8(np.array(mask)) * 255
        contours, _ = cv2.findContours(mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.drawContours(image_array, contours, -1,EDGE_COLOR, EDGE_WIDTH)
        image_i=Image.fromarray(contour_image)

        return image_i
    
    def blur_circle_method(self, img, box, image_name=None, box_index=None, **kwargs):
        image=self.blur_method(img,box,image_name,box_index)
        return self.circle_method(image,box)
    
    def gray_circle_method(self, img, box, image_name=None, box_index=None, **kwargs):
        image=self.gray_method(img,box,image_name,box_index)
        return self.circle_method(image,box)


    def __call__(self, out_path: str = None, methods: List[str] = ['blur'], enlarge_boxes: float = 0, is_dynamic: bool = True, resize_square: bool = True, fine_grained: bool = True, sam_cache_path: str = None, sam_temp_path: str = None, sam_iter: int = 0, sam_method: int = 0) -> IMAGE_CLASS:
        self.enlarge_boxes=enlarge_boxes
        self.is_dynamic=is_dynamic
        self.fine_grained=fine_grained
        self.sam_iter=sam_iter
        self.sam_method=sam_method

        method_map={
            'blur': self.blur_method,
            'crop': self.crop_method,
            'circle': self.circle_method,
            'gray': self.gray_method,
            'edge': self.edge_method,
            'blur-circle': self.blur_circle_method,
            'gray-circle': self.gray_circle_method
        }

        if out_path is None:
            raise ValueError("empty out path")
        if os.path.exists(out_path) is None:
            os.makedirs(out_path)
        assert all (method in ['blur','crop','circle','gray','edge','blur-circle','gray-circle'] for method in methods)
        if(fine_grained):
            self.sam_cache_path=sam_cache_path
            self.fpvg=FPVG(sam_cache_path=sam_cache_path,sam_temp_path=sam_temp_path)
        img_dict={}
        for datum in tqdm(self.data):
            file_name = datum["file_name"]
            img_path = os.path.join(self.image_path, file_name)
            img_dict[file_name]={}
            
            image = Image.open(img_path).convert('RGB')
            image_name = str(datum["image_id"])
            
            boxes = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in self.detections_map[int(datum["image_id"])]]
            if len(boxes) == 0:
                boxes = [Box(x=0, y=0, w=image.width, h=image.height)]
            boxes_in_range = [[
                max(boxes[i].left, 0),
                max(boxes[i].top, 0),
                min(boxes[i].right, image.width),
                min(boxes[i].bottom, image.height)
            ] for i in range(len(boxes))]

            out_img_path=os.path.join(out_path,file_name)

            for method in methods:
                method_path =os.path.join(out_img_path,method)
                img_dict[file_name][method]={}
                if not os.path.exists(method_path):
                    os.makedirs(method_path)
                for i in range(len(boxes)):
                    box = boxes_in_range[i]
                    img_path = os.path.join(method_path,str(box)+'.jpg')
                    if not os.path.exists(img_path):
                        image_i = method_map[method](
                            img=image, 
                            box=box,
                            image_name=image_name, 
                            box_index=i
                        )
                        # 多个视觉模型encode的cropped一个物体
                        if resize_square:
                            image_i = self.resize2square(image_i)
                        image_i.save(img_path)
                    img_dict[file_name][method][str(box)]=Image.open(img_path)            
        return IMAGE_CLASS(image_dict=img_dict)
        
if __name__ == '__main__':
    processer=IMAGE_PREPROCESS(image_path='data/testa/images', input_file='data/testa/annos.jsonl', detector_file_path='data/dets_dict.json')
    processer(out_path='out_img_new_sam',methods=['blur','crop','circle','gray','edge','blur-circle','gray-circle'],enlarge_boxes=0,is_dynamic=True,resize_square=True,sam_cache_path='sam_cache_aotu', sam_temp_path='sam_tmp', sam_iter=0, sam_method=0)
