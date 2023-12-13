
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator, utils
import os
import cv2
import random
from typing import Literal, List, Tuple
import pickle

class FPVG:
    def __init__(self, sam_cache_path=None, sam_temp_path=None) -> None:
            
        if sam_cache_path is not None and not os.path.exists(sam_cache_path):
            # raise ValueError("sam cache path error")
            os.makedirs(sam_cache_path)
        self.cache_path=sam_cache_path
        if sam_temp_path is not None and not os.path.exists(sam_temp_path):
            # raise ValueError("sam cache path error")
            os.makedirs(sam_temp_path)
        self.sam_temp_path=sam_temp_path

        sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.mask_generator=SamAutomaticMaskGenerator(model=self.sam)
        # , pred_iou_thresh=0.75, points_per_side=50, points_per_batch=60, stability_score_thresh=0.96
        
    # def __get_f1_score():


    def __get_mask(self, pic, box: list = None, point_coords: list = None, point_labels: list = None, iter: int = 0) -> np.ndarray:
        img_array = np.array(pic)
        self.predictor.set_image(img_array)
        if point_coords is not None:
            point_coords=np.array(point_coords)
            point_labels=np.array(point_labels)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box = np.array(box),
            multimask_output=True,
        )
        mask = np.uint8(masks[0]) * 255

        iter_i=0
        while iter_i<iter:
            iter_i+=1
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # contour_image = cv2.drawContours(pic_array, contours, -1, (0, 255, 0), 3)
            
            points=[]
            lables=[]
            for p in contours:
                num=int(len(p)*0.1)
                p=random.sample(list(p),num)
                for p1 in p:
                    for pp in p1:
                        points.append(pp)
                        lables.append(1)
                        img_array=cv2.circle(img_array, pp, 1, (0, 0, 255), 4)
            # contour_image=img_array

            # cv2.imwrite(f'b{iter_i}.jpg',contour_image)

            if len(points) > 0:
                points=np.array(points)
                lables=np.array(lables)
            else:
                points=None
                lables=None
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=lables,
                box = np.array(box),
                multimask_output=True,
            )
            mask1 = np.uint8(masks[0]) * 255
            # cv2.imwrite(f'c{iter_i}.jpg',mask1)

            # 使用逻辑OR操作合并两张图像
            mask = np.logical_or(mask, mask1).astype(np.uint8)*255
            # cv2.imwrite(f'a{iter_i}.jpg',mask)
        return mask

    def __get_scored_masks(self, pic, pic_name, box: list = None) -> List[Tuple[float, np.ndarray]]:
        img_array = np.array(pic)
        if self.sam_temp_path is not None:
            tmp=os.path.join(self.sam_temp_path,pic_name+'.json')
            if os.path.exists(tmp):
                tmp_file=open(tmp,'rb')
                masks=pickle.load(tmp_file)
                tmp_file.close()
            else:
                masks=self.mask_generator.generate(img_array)
                tmp_file=open(tmp,'wb')
                pickle.dump(masks,tmp_file)
                tmp_file.close()
        else:
            masks=self.mask_generator.generate(img_array)

        scored_masks=[]
        for mask_dict in masks:
            box_mask=[
                mask_dict["bbox"][0],
                mask_dict["bbox"][1],
                mask_dict["bbox"][0]+mask_dict["bbox"][2],
                mask_dict["bbox"][1]+mask_dict["bbox"][3]
            ]
            if box[2]<box_mask[0] or box[3]<box_mask[1] or box_mask[2]<box[0] or box_mask[3]<box[1]:
                intersection_area=0
            else:
                x=[box_mask[0],box_mask[2],box[0],box[2]]
                y=[box_mask[1],box_mask[3],box[1],box[3]]
                x.sort()
                y.sort()
                intersection_box=[
                    x[1],
                    y[1],
                    x[2],
                    y[2]
                ]
                intersection_area=(intersection_box[3]-intersection_box[1])*(intersection_box[2]-intersection_box[0])
            box_area=(box[3]-box[1])*(box[2]-box[0])
            bbox_area=mask_dict["bbox"][2]*mask_dict["bbox"][3]

            recall_score=intersection_area/box_area
            acc_score=intersection_area/bbox_area

            BETA=1
            f_score=(1+BETA**2)*(acc_score*recall_score)/(BETA**2*acc_score+recall_score+1e-10)

            
            # print(acc_score,recall_score,f_score)
            scored_masks.append((f_score, mask_dict['segmentation']))
        
        scored_masks.sort(key=lambda i:i[0], reverse=True)
        return scored_masks

    def get_mask(self, pic: Image, image_name: str, box_index: int, box: list = None, iter: int = 0, method: int = 0) -> Image: # 0: generate 1:predict
        if self.cache_path is not None:
            dir_path=os.path.join(self.cache_path, image_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            name='_'.join(map(str,box))
            if iter>0:
                name+=f'_{iter}'
            # if mid_point:
            #     name+='_mid'
            #     if point_coords is None:
            #         point_coords=[]
            #         point_labels=[]
            #     point_coords.append([box[0]+box[2],box[1]+box[3]])
            #     point_labels.append(1)
            mask_path=os.path.join(dir_path, name+'.jpg')
            if os.path.exists(mask_path):
                return Image.open(mask_path).convert('1')

            if method == 0:
                # masks=self.__get_scored_masks(pic, pic_name=image_name, box=box)
                # mask_l=list(map(lambda i:i[3],masks))
                # percent_box_l=list(map(lambda i:i[1],masks))
                # percent_mask_l=list(map(lambda i:i[2],masks))
                # mask=mask_l[0]
                # percent=percent_box_l[0]
                # i = 0
                # while i < len(percent_mask_l):
                #     # percent <= 0.5 and 
                #     if percent_mask_l[i] >= 0.90: 
                #         percent += percent_box_l[i]
                #         mask = np.logical_or(mask, mask_l[i]).astype(np.uint8)*255
                #     i += 1
              
                masks=self.__get_scored_masks(pic, pic_name=image_name, box=box)
                score, mask=masks[0]
                # print(score)

            elif method == 1:
                mask=self.__get_mask(pic, box=box, iter = iter)
            mask=Image.fromarray(mask).convert('1')
            mask.save(mask_path)
            return mask
        
        if method == 0:
            # masks=self.__get_scored_masks(pic, pic_name=image_name, box=box)
            # maskl=list(map(lambda i:i[2],masks))
            # percentl=list(map(lambda i:i[1],masks))
            # mask=maskl[0]
            # percent=percentl[0]
            # i = 1
            # while i < len(percentl):
            #     percent += percentl[i]
            #     mask = np.logical_or(mask, maskl[i]).astype(np.uint8)*255
            #     i += 1
            masks=self.__get_scored_masks(pic, pic_name=image_name, box=box)
            score, mask=masks[0]
                # print(score)
        elif method == 1:
            mask=self.__get_mask(pic, box=box, iter = iter)
        return Image.fromarray(mask).convert('1')
            
            
# if __name__=='__main__':
#     fp=FPVG(None)
#     pic=Image.open('data/val/images/000000567964.jpg')
#     pic.save('m.jpg')
#     box=[0.0, 278.4039001464844, 305.285400390625, 640.0]
#     mask=fp.get_mask(pic,'1',1,box,0,0)
#     inner_circle = pic.copy()
#     outer_circle = pic.copy()
#     # 对圆外部的部分进行高斯模糊处理
#     outer_circle = outer_circle.filter(ImageFilter.GaussianBlur(radius=100))  # 调整radius以控制模糊程度
#     # 合并两部分图像，只对圆外部进行模糊
#     result_image = Image.composite(inner_circle, outer_circle, mask)
#     result_image.save('d.jpg')


#     # sam_checkpoint = "./sam/sam_vit_h_4b8939.pth"
#     # model_type = "vit_h"
#     # device = "cuda"
#     # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     # sam.to(device=device)
#     # print("开始")
#     # box=[0.0, 278.4039001464844, 305.285400390625, 640.0]
#     # # p=np.linspace(box[0],box[2],num=10)
#     # # p2=np.linspace(box[1],box[3],num=10)
#     # # points=np.meshgrid(p,p2)

#     # mask_generatoe=SamAutomaticMaskGenerator(model=sam)
#     # pic=Image.open('data/val/images/000000567964.jpg')
#     # pic.save('m.jpg')
    
#     # img_array = np.array(pic)
#     # masks=mask_generatoe.generate(img_array)
#     # scored_masks=[]
#     # for mask_dict in masks:
#     #     box_mask=[
#     #         mask_dict["bbox"][0],
#     #         mask_dict["bbox"][1],
#     #         mask_dict["bbox"][0]+mask_dict["bbox"][2],
#     #         mask_dict["bbox"][1]+mask_dict["bbox"][3]
#     #     ]
#     #     if box[2]<box_mask[0] or box[3]<box_mask[1] or box_mask[2]<box[0] or box_mask[3]<box[1]:
#     #         intersection_area=0
#     #     else:
#     #         x=[box_mask[0],box_mask[2],box[0],box[2]]
#     #         y=[box_mask[1],box_mask[3],box[1],box[3]]
#     #         x.sort()
#     #         y.sort()
#     #         intersection_box=[
#     #             x[1],
#     #             y[1],
#     #             x[2],
#     #             y[2]
#     #         ]
            
#     #         print(box,box_mask,intersection_box)
#     #         intersection_area=(intersection_box[3]-intersection_box[1])*(intersection_box[2]-intersection_box[0])
        
#     #     box_area=(box[3]-box[1])*(box[2]-box[0])
#     #     bbox_area=mask_dict["bbox"][2]*mask_dict["bbox"][3]
#     #     recall_score=intersection_area/box_area
#     #     acc_score=intersection_area/bbox_area
#     #     BETA=1
#     #     f_score=(1+BETA**2)*(acc_score*recall_score)/(BETA**2*acc_score+recall_score+1e-10)
#     #     print(acc_score,recall_score,f_score)
#     #     scored_masks.append((f_score,mask_dict['segmentation']))
    
#     # scored_masks.sort(key=lambda i:i[0], reverse=True)
#     # for score,mask in scored_masks:
#     #     mask=mask.copy().astype(np.uint8)*255
#     #     # print(score)
#     #     mask=Image.fromarray(mask,'L')
#     #     image_i = pic.copy()
        
#     #     inner_circle = image_i.copy()
#     #     outer_circle = image_i.copy()
#     #     # 对圆外部的部分进行高斯模糊处理
#     #     outer_circle = outer_circle.filter(ImageFilter.GaussianBlur(radius=100))  # 调整radius以控制模糊程度
#     #     # 合并两部分图像，只对圆外部进行模糊
#     #     result_image = Image.composite(inner_circle, outer_circle, mask)

#     #     result_image.convert('RGB').save(f'sam_test_pics/{score}.jpg')

#     # masks1=mask_generatoe.postprocess_small_regions(masks,1,0.01)

#     # fp=FPVG(None)
#     # pic=Image.open('data/val/images/000000567964.jpg')
#     # pic.save('m.jpg')
#     # box=[0.0, 278.4039001464844, 305.285400390625, 640.0]
#     # img_array = np.array(pic)
#     # fp.predictor.set_image(img_array)

#     # masks, scores, logits = fp.predictor.predict(
#     #     point_coords=np.array([[305.285400390625/2,(278.4039001464844+640.0)/2]]),
#     #     point_labels=np.array([1]),
#     #     box = np.array(box),
#     #     multimask_output=True,
#     # )
#     # for i,mask in enumerate(masks):
#     #     mask=np.uint8(masks[0]) * 255
#     #     cv2.imwrite(f'm{i}.jpg',mask)
    
#     # # mask.save('c1.jpg')
#     # # mask=Image.open('sam_cache/1488/0.0_33.2506103515625_65.36231231689453_318.8167419433594.jpg').convert('1')

#     # # inner_circle = pic.copy()
#     # # outer_circle = pic.copy()
#     # # # 对圆外部的部分进行高斯模糊处理
#     # # outer_circle = outer_circle.filter(ImageFilter.GaussianBlur(radius=100))  # 调整radius以控制模糊程度
#     # # # 合并两部分图像，只对圆外部进行模糊
#     # # result_image = Image.composite(inner_circle, outer_circle, mask)
#     # # result_image.save('d.jpg')
#     print("结束")
#     exit(0)
