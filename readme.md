# README

## 命令说明

### 如何运行

1. 安装依赖：

```bash
pip install -r requirements.txt
```

将sam模型下载至 `models/` 文件夹下

2.  运行

调用shell脚本运行，生成结果（内部为运行main.py的指令）：

``` bash
bash run.sh
```

若不想通过调用shell脚本运行，或要更改参数，通过输入以下命令直接运行main.py:

``` bash
CUDA_VISIBLE_DEVICES=0 python main.py \
      --input_file 'data/testa/annos.jsonl' \
      --detector_file 'data/dets_dict.json' \
      --image_root 'data/testa/images' \
      --clip_model 'ViT-B/32,RN50x16' \
      --results_path 'result/result_testa.json' \
      --cache_path 'cache/test' \
      --box_representation_method 'blur,circle,crop,gray,edge,blur-circle,gray-circle' \
      --method 'baseline' \
      --device 0 \
      --dynamic_color_width True \
      --fine_grained True \
      --sam_cache_path 'cache/sam' \
      --sam_temp_path 'cache/sam_tmp' \
      --resize_square True \
      --enlarge_boxes 0 \
      --size_Q 300 \
      --box_area_threshold 0.06 \
      --img_cache_path 'cache/img'
```

### 问题说明

运行时，为了加速计算过程并减少内存占用，会自动生成`/cache/`文件夹，用于存放cache文件，其占用硬盘空间较大，运行结束可以删除。由于cache文件的存在，若运行中由于内存或显存不足进程中止，可以直接重新运行(不要删除cache)，程序将从中止的位置继续运行。

运行结束后，结果存放在自动生成的`/result/`文件夹中，名称为`xxx.json`。

由于算法原因，运行过程中需要用到较大内存、显存以及硬盘空间，需要主机用于至少60GB内存和48GB显存。

以下为可能遇到的问题：

- 运行时突然停止，显示`kill`

这是由于内存不足导致，可以直接重新运行(不要删除cahce)。若需要一次运行结束，可以考虑更换更大内存的主机运行。

- 运行时突然停止，显示`RuntimeError: CUDA out of memory. Tried to allocate 916.00 MiB (GPU 0; 6.00 GiB total capacity; 4.47 GiB already allocated; 186.44 MiB free; 4.47 GiB reserved in total by PyTorch)`(具体数值可能不同)

这是由于显存不足导致，可以直接重新运行(不要删除cahce)。若需要一次运行结束，考虑更换安装有更大显存的显卡运行，或将运行参数中`--size_Q`大小更改为100或50，但这会导致正确率下降。

## 流程介绍

1. 加载`main.py`程序

      主程序的运行逻辑如下：
      * 首先加载我们模型运行的一些参数。
      * 加载图片类型数据并调用`IMAGE_PREPROCESS`处理。
      * 设置相关参数。
      * 加载`{ 图片id（int）: 检测框position（list）}`类型数据。
      * 对数据集进行分割。
      * 调用`executor`进行推理。

2. `image_preprocess.py`的实现以及功能

      此代码实现多种对图像进行预处理的方法，其中实现了以下的一些函数。

      ```python
      # 动态改变圈的宽度和颜色方法 以及 格式转换
      dynamic_color_width(self, box: Box, image: Image, max_width = 7, min_width = 3) -> (tuple, int):
      resize2square(self, img: Image.Image) -> Image.Image:
      # 下面是我们所创建的7个图像处理方法：
      crop_method(self, img, box, **kwargs):
      circle_method(self, img, box, **kwargs):
      blur_method(self, img, box, image_name=None, box_index=None, **kwargs):
      gray_method(self, img, box, image_name=None, box_index=None, **kwargs):
      edge_method(self, img, box, image_name=None, box_index=None, **kwargs):
      blur_circle_method(self, img, box, image_name=None, box_index=None, **kwargs):
      gray_circle_method(self, img, box, image_name=None, box_index=None, **kwargs):
      # 调用函数
      __call__(self, out_path: str = None, methods: List[str] = ['blur'], enlarge_boxes: float = 0, is_dynamic: bool = True, resize_square: bool = True, fine_grained: bool = True, sam_cache_path: str = None, sam_temp_path: str = None, sam_iter: int = 0, sam_method: int = 0) -> IMAGE_CLASS:
      ```

3. `FPVG.py`的实现以及功能

      此代码实现`get mask`的功能，为`image_preprocess`做铺垫，其中主要实现了以下的函数。

      ```python
      # 通过单个检测框生成 mask 的方法 （实际使用的是后者的整体检测框生成）
      __get_mask(self, pic, box: list = None, point_coords: list = None, point_labels: list = None, iter: int = 0) -> np.ndarray:
      # 通过整体检测框生成 mask 并使用 F1_score 计算得分
      __get_scored_masks(self, pic, pic_name, box: list = None) -> List[Tuple[float, np.ndarray]]:
      # 通过得分的排序选择出 the best mask 
      get_mask(self, pic: Image, image_name: str, box_index: int, box: list = None, iter: int = 0, method: int = 0) -> Image:
      ```

4. `executor.py`的实现以及功能

      此代码为模型核心部分，主要调动各种资源完成推理，其中主要实现了以下几个关键函数。
      
      ```python
      # 下面两个函数是对文本端的处理（后者为增强版，实际使用为前者）
      preprocess_text(self, text: str) -> torch.Tensor:
      preprocess_text_v1(self, text:str, images, boxes, caption_hash, box_representation_methods, image_name, model_name, images_t, model) -> torch.Tensor:
      # 关键的模型算法
      _Q(self, image_features, model, model_name) -> torch.Tensor:
      # 调用 clip 的 API（后者为增强版，实际使用为后者）
      call_model(self, model: torch.nn.Module, images: torch.Tensor, text: torch.Tensor, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
      call_model_batch_text(self, model, images = None, texts = None, image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
      ```

## 参数

以下为主要的输入参数：

```
      --input_file 输入文件路径(默认)
      --detector_file 输入框路径(默认)
      --image_root 图片路径(默认)
      --clip_model clip模型路径(默认)
      --results_path 输出结果位置
      --cache_path cache文件存放路径
      --box_representation_method 检测框在图片中的呈现方法
      --method 计算检测框得分方法
      --device 使用设备(0为GPU)
      --dynamic_color_width 呈现检测框时使用动态颜色粗细
      --fine_grained 使用SAM模型得到具体检测物体
      --sam_cache_path sam模型对检测框内容分割结果存放路径
      --sam_temp_path sam模型中间结果文件存放路径(占用内存较大)
      --resize_square 将输入图像填充灰色变成正方形
      --enlarge_boxes 扩大检测框尺寸
      --size_Q 计算图片均值数量
      --box_area_threshold 检测框最小尺寸
      --img_cache_path 处理后图片cache路径

```

## 算法

1. 对于每张图片及其中的每个检测框，生成一系列新图片，图片数量取决于呈现方法个数(box_representation_method)，现有7种方法：blur,circle,crop,gray,edge,blur-circle,gray-circle。此外，如果启用了fine_grained(--fine_grained true)，那么blur,gray,blur-circle,gray-circle这4个方法处理方式会有所改变。
    - blur: 不启用fine_grained：将检测框以外的部分模糊。启用fine_grained：将将检测框中的主体部分以外的部分模糊。
    - circle: 用圆圈标注检测框位置。
    - crop: 将检测框部分裁切出来。
    - gray: 不启用fine_grained：将检测框以外的部分变成黑白。启用fine_grained：将将检测框中的主体部分以外的部分变成黑白。
    - edge: (fine_grained必须为true)将检测框中的主体部分描边。
    - blur-circle: 不启用fine_grained：将检测框以外的部分模糊，再用圆圈标注检测框位置。启用fine_grained：将检测框中的主体部分以外的部分模糊，再用圆圈标注检测框位置。
    - gray-circle: 不启用fine_grained：将检测框以外的部分变成黑白，再用圆圈标注检测框位置。启用fine_grained：将检测框中的主体部分以外的部分变成黑白，再用圆圈标注检测框位置。
如果启用了resize_square，这些图片还会通过填充灰色变为正方形。检测框的最终得分为：这些图片与文本得分的总和。

2. 抽取一系列文本(数量为--size_Q)与每个检测框计算得分，作为这个检测框的偏差，在计算检测框得分时减去它的偏差。越大效果越好，但需要更多显存，均衡后最佳值为300。

3. 分析句子中的主语与谓语，构建空间关系。

## 数据集

数据集下载链接：  

相关说明详见 `data/readme.md`