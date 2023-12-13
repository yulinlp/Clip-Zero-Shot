from typing import List, Dict, Union, Tuple

from PIL import Image, ImageDraw, ImageFilter
from matplotlib import pyplot as plt
import hashlib
import os

import torch
import torchvision.transforms as transforms
import clip
from transformers import BertTokenizer, RobertaTokenizerFast

from .interpreter import Box

from .image_preprocess import IMAGE_CLASS

class Executor:
    def __init__(self, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: float = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, fine_grained: bool = False, sam_cache_path: str = None, sam_iter: int = 0, resize_square: bool = False, cache_path: str = None, img_dict: IMAGE_CLASS = None, corpus_Q = None) -> None:
        
        IMPLEMENTED_METHODS = ["crop", "shade", "circle", "blur", "gray","edge","blur-circle","gray-circle"]
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        # box_representation_method(将检测框图像表示成单独一个图像的方式) 默认是'裁剪,模糊'
        self.box_representation_method = box_representation_method
        # box_method_aggregator(组合检测框图像表示分数的方法) 默认是求和
        self.method_aggregator = method_aggregator
        self.enlarge_boxes = enlarge_boxes
        self.device = device
        self.expand_position_embedding = expand_position_embedding
        # square_size(是不是正方形)
        self.square_size = square_size
        # blur_std_dev(高斯模糊的标准差) 论文中测试用的是100
        self.blur_std_dev = blur_std_dev
        # fine_grained(使用细粒度划分(sam))
        # self.fine_grained = fine_grained
        # if(fine_grained):
        #     self.sam_cache_path=sam_cache_path
        #     # self.fpvg=FPVG(sam_cache_path=sam_cache_path)
        #     self.sam_iter=sam_iter
        # self.resize_square=resize_square
        # self.GRAY = (69,69,69)
        self.cache_path = cache_path
        self.img_dict=img_dict
            
    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        # 使用图像模型进行encode
        # 可能有多个模型进行encode，返回列表
        return [preprocess(image) for preprocess in self.preprocesses]

    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_inputs(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, file_name: str = None, is_dynamic: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # 输入是大图像，三种方式处理，输出encode后的每个检测框物体的tensor
        images = []
        
        for preprocess in self.preprocesses:
            images.append([])
        if self.cache_path is None or any([not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for model_name in self.model_names for method_name in self.box_representation_method.split(',')]):
            image_dict=self.img_dict.get_image_dict(file_name)
            boxes_in_range = [[
                max(boxes[i].left, 0),
                max(boxes[i].top, 0),
                min(boxes[i].right, image.width),
                min(boxes[i].bottom, image.height)
            ] for i in range(len(boxes))]
            box_representation_methods = self.box_representation_method.replace(' ','').split(',')
            for method in box_representation_methods:
                for i in range(len(boxes)):
                    box = boxes_in_range[i]
                    preprocessed_images = self.preprocess_image(image_dict[method][str(box)])
                    for j, img in enumerate(preprocessed_images):
                        # j是第j个视觉模型
                        images[j].append(img.to(self.device))
            # 将多个视觉模型encode的结果合一
            imgs = [torch.stack(image_list) for image_list in images]
        else:
            imgs = [[] for _ in self.models]
        
        return imgs

    def load_image_text_cache(self, model_name, boxes, caption_hash, box_representation_methods, image_name):
        if self.cache_path is not None:
            text_cache_path = os.path.join(self.cache_path, model_name, "text")
        image_features = None
        text_features = None
        if self.cache_path is not None and os.path.exists(os.path.join(self.cache_path, model_name)):
            if os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")):
                text_features = torch.load(os.path.join(text_cache_path, caption_hash+".pt"), map_location=self.device)
                # print("加载text cache成功")
            if os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                # print(f"正在加载{image_name}的image_features...")
                if all([os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for method_name in box_representation_methods]):
                    image_features = []
                    for method_name in box_representation_methods:
                        features = torch.load(os.path.join(self.cache_path, model_name, image_name, method_name+".pt"), map_location=self.device)
                        # print("features: ", features.keys())
                        image_features.append(torch.stack([
                            features[(int(box.x), int(box.y), int(box.w), int(box.h))]
                            for box in boxes
                        ]))
                    image_features = torch.stack(image_features)
                    image_features = image_features.view(-1, image_features.shape[-1])
                # print("加载image cache成功")       
        return image_features, text_features
    
    @torch.no_grad()
    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, file_name: str = None, is_dynamic: bool = False) -> torch.Tensor:
        images = self.tensorize_inputs(caption, image, boxes, image_name, file_name, is_dynamic)
        
        text_tensor = self.preprocess_text(caption.lower()).to(self.device)
        all_logits_per_image = []
        all_logits_per_text = []
        box_representation_methods = self.box_representation_method.split(',')
        caption_hash = hashlib.md5(caption.encode('utf-8')).hexdigest()
        for model, images_t, model_name in zip(self.models, images, self.model_names):
            # load cache
            # print(len(images_t))
            # print(boxes)
            image_features, text_features = self.load_image_text_cache(model_name, boxes, caption_hash, box_representation_methods, image_name)
            # if self.text_prompt == True and text_features is None:
            #     text_features = self.preprocess_text_v1(caption.lower(), images, boxes, caption_hash, box_representation_methods, image_name, model_name, images_t, model)
            logits_per_image, logits_per_text, image_features, text_features = self.call_model(model, images_t, text_tensor, image_features=image_features, text_features=text_features)
            # corpus_Q
            if self.corpus_Q_tensors is not None:
                bias = self._Q(image_features = image_features, model = model, model_name=model_name)
                logits_per_text -= bias
                
            all_logits_per_image.append(logits_per_image)
            all_logits_per_text.append(logits_per_text)
            # save cache
            if self.cache_path is not None and image_name is not None and image_features is not None:
                image_features = image_features.view(len(box_representation_methods), len(boxes), image_features.shape[-1])
                if not os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    os.makedirs(os.path.join(self.cache_path, model_name, image_name))
                for i in range(image_features.shape[0]):
                    method_name = box_representation_methods[i]
                    if not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")):
                        image_features_dict = {(int(box.x), int(box.y), int(box.w), int(box.h)): image_features[i,j,:].cpu() for j, box in enumerate(boxes)}
                        # print("image_features_dict: ", image_features_dict.keys())
                        torch.save(image_features_dict, os.path.join(self.cache_path, model_name, image_name, method_name+".pt"))
            if self.cache_path is not None:
                text_cache_path = os.path.join(self.cache_path, model_name, "text")
            if self.cache_path is not None and not os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")) and text_features is not None:
                assert text_features.shape[0] == 1
                if not os.path.exists(text_cache_path):
                    os.makedirs(text_cache_path)
                torch.save(text_features.cpu(), os.path.join(text_cache_path, caption_hash+".pt"))
            
        all_logits_per_image = torch.stack(all_logits_per_image).sum(0)
        all_logits_per_text = torch.stack(all_logits_per_text).sum(0)
        if self.method_aggregator == "max":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
        all_logits_per_text = all_logits_per_text.view(-1)
        
        return all_logits_per_text

class ClipExecutor(Executor):
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: float = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, fine_grained: bool = False, sam_cache_path: str = None, sam_iter: int = 0, resize_square: bool = False, cache_path: str = None, img_dict: IMAGE_CLASS = None, corpus_Q = None, text_prompt = False) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, fine_grained, sam_cache_path, sam_iter, resize_square, cache_path, img_dict, corpus_Q)
        self.clip_models = clip_model.split(",")
        self.model_names = [model_name.replace("/", "_") for model_name in self.clip_models]
        self.models = []
        self.preprocesses = []        
        for model_name in self.clip_models:
            # 最官方的加载模型方式，cache没有pt，会自动下载
            model, preprocess = clip.load(model_name, device=device, jit=False)
            self.models.append(model)
            if self.square_size:
                print("Square size!")
                # 调整输入图片的大小
                preprocess.transforms[0] = transforms.Resize((model.visual.input_resolution, model.visual.input_resolution), interpolation=transforms.InterpolationMode.BICUBIC)
            self.preprocesses.append(preprocess)
        self.models = torch.nn.ModuleList(self.models)
        
        self.corpus_Q_tensors = None
        self.corpus_Q_features = {}
        # corpus_Q 初始化
        if corpus_Q is not None:
            self.corpus_Q_tensors = clip.tokenize(corpus_Q).to(self.device) # embedding的corpus_Q
            for model_name, model in zip(self.model_names, self.models):
                text_features = model.encode_text(self.corpus_Q_tensors)
                self.corpus_Q_features[model_name] = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_prompt = False
        print("是否启用text_prompt? ", self.text_prompt)

    def preprocess_text(self, text: str) -> torch.Tensor:
        # 加prompt
        return clip.tokenize(["this is a photo of "+text.lower()])

    def preprocess_text_v1(self, text:str, images, boxes, caption_hash, box_representation_methods, image_name, model_name, images_t, model) -> torch.Tensor:
        # load prompt pool
        prompt_pool = ['satellite imagery of {}' ,'this is a photo of {}' ,'aerial view of a {}' ,'i love my {}' ,'a drawing of the {}' ,'a video of a person {}' ,'satellite photo of {}' ,'a photo of the person performing {}' ,'there are {} shapes' ,'a video of the person using {}' ,'a centered satellite photo of {}' ,'a example of the person doing {} ' ,'a photo of a person practicing {} ' ,'a example of the person performing {} ' ,'art of a {} ' ,'a {} ' ,'itap of the {} ' ,'a drawing of a {} ' ,'a origami {} ' ,'a video of {}', 'a photo of a nice {} ' ,'a blurry photo of a {} ' ,'they look {} ' ,'the {} in a video game ' ,'a face that looks {} ' ,'a picture of {} objects ' ,'a close-up photo of the {} ' ,'a photo of {} ' ,'a photo i took in {} ' ,'a example of the person during {} ' ,'a centered satellite photo of the {} ' ,'a street sign with the number {}' ,'a photo of a clean {} ' ,'a photo of a weird {} ' ,'a photo of a small {} ' ,'a high contrast photo of a {} ' ,'the nearest shape in this image is {} ' ,'a photo of the large {} ' ,'an example of {}' ,'a pixelated photo of the {} ' ,'a histopathology slide showing {}' ,'a embroidered {} ' ,'satellite view of a {} ' ,'a high contrast photo of the {} ' ,'a photo of the {} texture ' ,'the closest shape in this rendered image is {} ' ,'a {} slide' ,'a demonstration of a person doing {} ' ,'a demonstration of a person practicing {} ' ,'this is a photo of {}' ,'a demonstration of the person using {} ' ,'a example of the person using {} ' ,'a photo of the person doing {} ' ,'a video of the person during {} ' ,'the number {} in the center of the image' ,'an example histopathological image showing {}' ,'a photo of the clean {} ' ,'a demonstration of the person practicing {} ' ,'the origami {} ' ,'the plushie {} ' ,'a photo of a {} thing ' ,'a photo of a cool {} ' ,'a sculpture of the {} ' ,'a example of a person during {} ' ,'a demonstration of the person {} ' ,'a low resolution photo of the {} ' ,'look at how {} they are', 'a photo of a person doing {} ' ,'a photo of the {} pattern ' ,'a bad photo of the {} ' ,'a {} texture' ,'the number {}' ,'aerial imagery of {} ' ,'a photo of a person {} ' ,'a jpeg corrupted photo of a {} ' ,'{} objects' ,'a photo of {} objects ' ,'a {} flower' ,'a rendition of the {} ' ,'a photo of the cool {} ' ,'{}' ,'a low resolution photo of a {} ' ,'{} shapes' ,'a photo from my home country of {} ' ,'a cropped photo of the {} ' ,'the plastic {} ' ,'a sculpture of a {} ' ,'a pixelated photo of a {} ' ,'itap of a {} ' ,'a demonstration of {} ' ,'a video of a person using {} ' ,'a doodle of a {} ' ,'a photo of the {} object ' ,'a sketch of a {} ' ,'a {} plant' ,'a satellite image of {} ' ,'a plastic {} ' ,'{} thing' ,'{} things' ,'a photo of the person using {} ' ,'itap of my {} ' ,'a example of a person using {}', 'the closest shape in this image is {} ' ,'a close-up photo of a {} ' ,'a bright photo of a {} ' ,'a photo of the person during {} ' ,'art of the {} ' ,'graffiti of the {} ' ,'a tattoo of a {} ' ,'a video of the person performing {} ' ,'a photo of a face looking {} ' ,'a sketch of the {} ' ,'aerial imagery of the {} ' ,'a dark photo of a {} ' ,'a tattoo of the {} ' ,'there are {} objects in the image ' ,'{}, an animal' ,'a photo of the dirty {} ' ,'a example of a person performing {} ' ,'a centered photo of a ”{}” traffic sign ' ,'a photo of the number: ”{}” ' ,'an overhead view of {} ' ,'a black and white photo of the {} ' ,'a zoomed in photo of a ”{}” traffic sign ' ,'a example of {} ' ,'a photo of a {} ' ,'a retinal image with {}' ,'a photo of the {}, a type of aircraft ' ,'a photo of a {} texture ' ,'a demonstration of a person during {} ' ,'a {} texture ' ,'a {} in a video game ' ,'a painting of the {} ' ,'a cropped photo of a {} ' ,'a demonstration of the person doing {} ' ,'a photo of a {} pattern ' ,'a example of a person practicing {} ' ,'a photo of a large {} ' ,'a photo from my visit to {} ' ,'an overhead image of {} ' ,'a photo of the weird {} ' ,'aerial photo of {} ' ,'satellite imagery of the {} ' ,'graffiti of a {} ' ,'a close up photo of a ”{}” traffic sign ' ,'a photo of a {}, a type of pet ' ,'a low contrast photo of a {} ' ,'a satellite photo of {} ' ,'a video of a person practicing {} ' ,'a demonstration of a person using {} ' ,'a painting of a {} ' ,'a cartoon {} ' ,'a photo of my new {} ' ,'aerial imagery of a {} ' ,'the cartoon {} ' ,'a low contrast photo of the {} ' ,'a photo of the big {} ' ,'a type of pet {}' ,'a video of the person {} ' ,'a video of a person performing {} ' ,'aerial view of the {} ' ,'a photo of a person during {}' ,'a photo of a {}, a type of aircraft ' ,'a video of a person during {} ' ,'a good photo of the {} ' ,'a photo of a {}, a type of bird ' ,'there are {} objects ' ,'a jpeg corrupted photo of the {} ' ,'a photo of the {} thing ' ,'a photo of a face showing the emotion: {} ' ,'a bad photo of a {} ' ,'a photo of the small {} ' ,'a picture of {} shapes ' ,'a centered satellite photo of a {} ' ,'a photo of a person using {} ' ,'aerial photo of a {} ' ,'a photo of a {}, a type of flower ' ,'a {} review of a movie ' ,'a rendering of the {} ' ,'a photo of a dirty {} ' ,'satellite imagery of a {} ' ,'a rendition of a {} ' ,'{} rotation' ,'photo of {} from the sky', 'a blurry photo of the {} ' ,'the toy {} ' ,'a video of a person doing {} ' ,'something at a {} rotation' ,'a photo of my clean {} ' ,'a example of a person {} ' ,'a demonstration of a person performing {} ' ,'the embroidered {} ' ,'aerial photo of the {} ' ,'a video of the person practicing {} ' ,'{} from above ' ,'a photo of the person practicing {} ' ,'a rendering of a {} ' ,'there are {} shapes in the image ' ,'a photo of a {} looking face ' ,'a rendered image of {} objects ' ,'an aerial view of {} ' ,'a photo of a big {} ' ,'a example of a person doing {} ' ,'an outdoor house number {}' ,'a photo of a hard to see {} ' ,'a dark photo of the {} ' ,'a example of the person {} ' ,'a demonstration of a person {} ' ,'a doodle of the {} ' ,'a good photo of a {} ' ,'an object rotated at {}' ,'a photo of the {} ' ,'a photo of many {} ' ,'a rendered image of {} shapes ' ,'histopathology image of {}' ,'a plushie {} ' ,'a photo i took while visiting {} ' ,'patient’s pathology examination indicates {}' ,'an outdoor number {} written on a sign' ,'a photo of the person {} ' ,'a photo showing the country of {} ' ,'a photo of a person performing {} ' ,'a photo of the nice {} ' ,'a demonstration of the person during {} ' ,'a bright photo of the {} ' ,'satellite view of the {} ' ,'a example of the person practicing {} ' ,'aerial view of {} ' ,'a photo of my old {} ' ,'a retina with {}' ,'a centered image of the number {}' ,'a fundus image with signs of {}' ,'an object located {}' ,'something rotated at {}' ,'satellite photo of a {} ' ,'a toy {} ' ,'a photo of a {} object ' ,'a video of the person doing {} ' ,'a photo of {}, a type of food ' ,'a photo of the hard to see {} ' ,'satellite photo of the {} ' ,'a photo of one {} ' ,'a photo of my dirty {} ' ,'a photo of my {} ' ,'a photo of the number {} written on a sign' ,'satellite view of {} ' ,'a demonstration of the person performing {} ' ,'a black and white photo of a {} ']
        # best_text_features = []
        # for model, images_t, model_name in zip(self.models, images, self.model_names):
        image_features, text_features = self.load_image_text_cache(model_name=model_name, boxes=boxes, caption_hash=caption_hash, box_representation_methods=box_representation_methods, image_name=image_name)
        captions = [prompt.format(text) for prompt in prompt_pool]
        caption_embeddings = clip.tokenize(captions).to(self.device)
        # print(caption_embeddings.shape)
        logits_per_caption, text_features = self.call_model_batch_text(model=model, images=images_t, texts=caption_embeddings, image_features=image_features, text_features=None)
        # print(logits_per_caption.shape)
        logits_per_caption = logits_per_caption.unsqueeze(1)
        if self.method_aggregator == "max":
            logits_per_caption = logits_per_caption.view(len(prompt_pool), -1, len(boxes)).max(dim=1, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            logits_per_caption = logits_per_caption.view(len(prompt_pool), -1, len(boxes)).sum(dim=1, keepdim=True)
        
        logits_per_caption = logits_per_caption.squeeze()
        if logits_per_caption.dim() == 1:
            print("维度调整")
            logits_per_caption = logits_per_caption.unsqueeze(dim=1)
        # print(logits_per_caption.shape)
        scores = logits_per_caption.max(dim=1)[0] - logits_per_caption.sum(dim=1) / logits_per_caption.size(dim=1)
        # print(scores.shape)
        scores = torch.softmax(scores, dim=0).unsqueeze(dim=0)
        # print(scores.shape, text_features.shape)
        # exit()
        best_text_feature = torch.mean(torch.matmul(scores, text_features), dim=0, keepdim=True)
        return best_text_feature

    def _Q(self, image_features, model, model_name) -> torch.Tensor:
        """return 正则项"""
        logits_per_caption, _ = self.call_model_batch_text(model=model, image_features=image_features, text_features=self.corpus_Q_features[model_name])
        logits_per_caption = logits_per_caption.unsqueeze(1)
        logits_per_caption = logits_per_caption.squeeze()
        # print(logits_per_caption.shape)
        if logits_per_caption.dim() == 1:
            # print("维度调整")
            logits_per_caption = logits_per_caption.unsqueeze(dim=1)
        # print(logits_per_caption.shape)
        average_tensor = logits_per_caption.mean(dim=0)
        # print(average_tensor.shape)
        return average_tensor
    
    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: torch.Tensor, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        # 图像/文本编码 -> 计算余弦相似度
        if image_features is None:
            # print('computing image features')
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            # print('computing text features')
            text_features = model.encode_text(text)
            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features
    
    def call_model_batch_text(self, model, images = None, texts = None, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        if image_features is None:
            # print('computing image features')
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if text_features is None:
            text_features = model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()  # [len(images), len(texts)]
        
        return logits_per_text, text_features
        

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, file_name: str = None, is_dynamic: bool = False) -> torch.Tensor:
        # 位置编码扩展
        if self.expand_position_embedding:
            original_preprocesses = self.preprocesses
            new_preprocesses = []
            original_position_embeddings = []
            for model_name, model, preprocess in zip(self.clip_models, self.models, self.preprocesses):
                if "RN" in model_name:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                else:
                    model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                new_preprocesses.append(transform)
                original_position_embeddings.append(original_positional_embedding)
            self.preprocesses = new_preprocesses
        # 不扩展，调用父类
        result = super().__call__(caption, image, boxes, image_name, file_name, is_dynamic)
        if self.expand_position_embedding:
            self.preprocesses = original_preprocesses
            for model, model_name, pos_embedding in zip(self.models, self.clip_models, original_position_embeddings):
                if "RN" in model_name:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(pos_embedding)
                else:
                    model.visual.positional_embedding = torch.nn.Parameter(pos_embedding)
        return result
    