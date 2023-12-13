from collections import defaultdict
import json
import argparse
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from code.interpreter import *
from code.executor import *
from code.methods import *
from code.image_preprocess import *
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')

METHODS_MAP = {
    "baseline": Parse,
}

def get_longest_sentences(size, sentences: List) -> List:
    return sorted(sentences, key=lambda sentence: len(sentence.split()), reverse=True)[:size]

def get_shortest_sentences(size, sentences: List) -> List:
    return sorted(sentences, key=lambda sentence: len(sentence.split()), reverse=False)[:size]
    
def get_middle_sentences(size, sentences: List) -> List:
    corpus = sorted(sentences, key=lambda sentence: len(sentence.split()), reverse=True)
    middle_index = len(corpus) // 2
    start_index = middle_index - size // 2
    end_index = start_index + size
    return corpus[start_index:end_index]

def get_most_frequency_sentences(size, sentences: List) -> List:
    # 使用Counter统计句子出现的次数
    sentence_counts = Counter(sentences)
    # 按照句子出现次数降序排序
    sorted_sentences = sentence_counts.most_common()
    # 获取出现次数最多的句子
    most_frequent_sentences = [sentence for sentence, count in sorted_sentences[:size]]
    return most_frequent_sentences

def get_most_frequency_sentences_hybrid(size_1, size_2, sentences):
    # 选择最短的 size_1 个句子
    shortest_sentences = sorted(list(set(sentences)), key=len)[:size_1]
    
    # 使用 Counter 统计最短句子在 sentences 中的频数
    sentence_counts = Counter(sentences)
    shortest_sentence_counts = [sentence_counts[sentence] for sentence in shortest_sentences]
    
    # 计算最短句子在结果中应该出现的次数
    total_count = sum(shortest_sentence_counts)
    sentence_ratios = [(sentence, count / total_count) for sentence, count in zip(shortest_sentences, shortest_sentence_counts)]
    
    # 根据句子的频数比例生成结果
    result = []
    while len(result) < size_2:
        for sentence, ratio in sentence_ratios:
            if len(result) < size_2:
                result.append(sentence)
            else:
                break    
    return result

def get_meaningful_sentences(sentences):
    meaningful_sentences = []
    
    # 遍历每个句子
    for sentence in sentences:
        # 利用Spacy进行语义分析
        doc = nlp(sentence)

        # 检查句子中是否包含有意义的词汇
        has_meaningful_words = any(token.is_alpha and not token.is_stop for token in doc)

        # 检查句子中是否包含数字
        has_digits = any(token.is_digit for token in doc)

        # 检查句子是否包含乱码
        has_gibberish = any(not token.is_alpha and not token.is_digit and not token.is_punct for token in doc)

        # 根据条件判断句子是否有意义
        if has_meaningful_words and not has_digits and not has_gibberish:
            meaningful_sentences.append(sentence)

    print(len(meaningful_sentences))
    return meaningful_sentences

def get_best_corpus(size_Q, data, strategy_id = 4) -> List:
    """ 
        1: 局部随机，全局随机
        2: 局部最长，全局最长
        3: 局部最短，全局最短
        4: 局部最长，全局中间
        5: 局部最短，全局中间   
        6: 出现频率最高
    """
    # 策略一：每个检测框对应的多个同义描述只选第一个，作为sampled_santences。随机选取size_Q个。
    # 59.9左右
    if strategy_id == 1:
        sampled_sentences = ['this is a photo of ' + random.sample(datum['sentences'], 1)[0] for datum in data]
        corpus_Q = random.sample(sampled_sentences, size_Q)
    # 策略二：每个检测框对应的多个同义描述选取字数最多的一个，作为sampled_sentences。选前size_Q个字数最多的。
    # 58.105
    elif strategy_id == 2:
        sampled_sentences = ['this is a photo of ' + get_longest_sentences(1, datum['sentences'])[0] for datum in data]
        corpus_Q = get_longest_sentences(size_Q, sampled_sentences)
    # 策略三：每个检测框对应的多个同义描述选取字数最少的一个，作为sampled_sentences。选前size_Q个字数最少的。
    # 60.485（未去重）
    elif strategy_id == 3:
        sampled_sentences = ['this is a photo of ' + get_shortest_sentences(1, datum['sentences'])[0] for datum in data]
        corpus_Q = get_shortest_sentences(size_Q, sampled_sentences)
    # 策略四：最优
    # 60.555（未清洗语料）
    elif strategy_id == 4:
        all_data = []
        for datum in data:
            all_data.extend([sentence.lower() for sentence in datum['sentences']])
        # all_data = get_meaningful_sentences(all_data)
        sampled_sentences = ['this is a photo of ' + sentence for sentence in all_data]
        corpus_Q = get_most_frequency_sentences_hybrid(size_1=size_Q // 2, size_2=size_Q, sentences=sampled_sentences)
    # 策略五：每个检测框对应的多个同义描述选取字数最少的一个，作为sampled_sentences。选size_Q个不多不少的。
    elif strategy_id == 5:
    # 60.393
        sampled_sentences = ['this is a photo of ' + get_shortest_sentences(1, datum['sentences'])[0] for datum in data]
        corpus_Q = get_middle_sentences(size_Q, sampled_sentences)
    elif strategy_id == 6:
    # 60.152
    # 60.471
        all_data = []
        for datum in data:
            all_data.extend([sentence.lower() for sentence in datum['sentences']])
        all_data = get_meaningful_sentences(all_data)
        sampled_sentences = ['this is a photo of ' + sentence for sentence in all_data]
        tmp_corpus_Q = get_most_frequency_sentences(3*size_Q, sampled_sentences)
        corpus_Q = get_shortest_sentences(size_Q, tmp_corpus_Q)
    print(corpus_Q[:50])
    print("生成corpus_Q成功!")
    return corpus_Q

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file with expressions and annotations in jsonlines format")
    parser.add_argument("--image_root", type=str, help="path to images")
    parser.add_argument("--clip_model", type=str, default="RN50x16,ViT-B/32", help="which clip model to use (should use RN50x4, ViT-B/32, or both separated by a comma")
    parser.add_argument("--albef_path", type=str, default=None, help="to use ALBEF (instead of CLIP), specify the path to the ALBEF checkpoint")
    parser.add_argument("--method", type=str, default="baseline", help="method to solve expressions")
    parser.add_argument("--box_representation_method", type=str, default="circle", help="method of representing boxes as individual images (crop, blur, or both separated by a comma)")
    parser.add_argument("--box_method_aggregator", type=str, default="sum", help="method of combining box representation scores")
    parser.add_argument("--box_area_threshold", type=float, default=0.0, help="minimum area (as a proportion of image area) for a box to be considered as the answer")
    parser.add_argument("--results_path", type=str, default='./output', help="(optional) output path to save results")
    parser.add_argument("--detector_file", type=str, default=None, help="(optional) file containing object detections. if not provided, the gold object boxes will be used.")
    parser.add_argument("--mock", action="store_true", help="(optional) mock CLIP execution.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use.")
    parser.add_argument("--shuffle_words", action="store_true", help="If true, shuffle words in the sentence")
    parser.add_argument("--gradcam_alpha", type=float, nargs='+', help="alpha value to use for gradcam method")
    parser.add_argument("--enlarge_boxes", type=float, default=0.0, help="(optional) whether to enlarge boxes when passing them to the model")
    parser.add_argument("--part", type=str, default=None, help="(optional) specify how many parts to divide the dataset into and which part to run in the format NUM_PARTS,PART_NUM")
    parser.add_argument("--batch_size", type=int, default=1, help="number of instances to process in one model call (only supported for baseline model)")
    parser.add_argument("--baseline_head", action="store_true", help="For baseline, controls whether model is called on both full expression and head noun chunk of expression")
    parser.add_argument("--mdetr", type=str, default=None, help="to use MDETR as the executor model, specify the name of the MDETR model")
    parser.add_argument("--albef_block_num", type=int, default=8, help="block num for ALBEF gradcam")
    parser.add_argument("--albef_mode", type=str, choices=["itm", "itc"], default="itm")
    parser.add_argument("--expand_position_embedding",action="store_true")
    parser.add_argument("--gradcam_background", action="store_true")
    parser.add_argument("--mdetr_given_bboxes", action="store_true")
    parser.add_argument("--mdetr_use_token_mapping", action="store_true")
    parser.add_argument("--non_square_size", action="store_true")
    parser.add_argument("--blur_std_dev", type=int, default=100, help="standard deviation of Gaussian blur")
    parser.add_argument("--gradcam_ensemble_before", action="store_true", help="Average gradcam maps of different models before summing over the maps")
    parser.add_argument("--cache_path", type=str, default=None, help="cache features")
    parser.add_argument("--size_Q", type=int, default=0, help="自己加的，公式自己看是啥意思")
    parser.add_argument("--dynamic_color_width", type=bool, default=False, help="是否动态改变圈的颜色和宽度")
    parser.add_argument("--fine_grained", type=bool, default=True, help="是否使用细粒度")
    parser.add_argument("--sam_cache_path", type=str, default=None, help="sam遮罩cache路径")
    parser.add_argument("--sam_temp_path", type=str, default=None, help="sam缓存文件cache路径")
    parser.add_argument("--sam_iter", type=int, default=0, help="sam迭代次数")
    parser.add_argument("--resize_square", type=bool, default=True, help="是否使用灰边")
    parser.add_argument("--img_cache_path", type=str, default=None, help="处理后图片cache路径")
    parser.add_argument("--text_prompt", type=bool, default=False, help="是否启用文本prompt tuning")
    # parser.add_argument("--size_text_prompt", type=int, default=0, help="用来算textprompt得分的图片集的大小")
    # Arguments related to Parse method.
    parser.add_argument("--no_rel", action="store_true", help="Disable relation extraction.")
    parser.add_argument("--no_sup", action="store_true", help="Disable superlative extraction.")
    parser.add_argument("--no_null", action="store_true", help="Disable null keyword heuristics.")
    parser.add_argument("--ternary", action="store_true", help="Disable ternary relation extraction.")
    parser.add_argument("--baseline_threshold", type=float, default=float("inf"), help="(Parse) Threshold to use relations/superlatives.")
    parser.add_argument("--temperature", type=float, default=1., help="(Parse) Sigmoid temperature.")
    parser.add_argument("--superlative_head_only", action="store_true", help="(Parse) Superlatives only quanntify head predicate.")
    parser.add_argument("--sigmoid", action="store_true", help="(Parse) Use sigmoid, not softmax.")
    parser.add_argument("--no_possessive", action="store_true", help="(Parse) Model extraneous relations as possessive relations.")
    parser.add_argument("--expand_chunks", action="store_true", help="(Parse) Expand noun chunks to include descendant tokens that aren't ancestors of tokens in other chunks")
    parser.add_argument("--parse_no_branch", action="store_true", help="(Parse) Only do the parsing procedure if some relation/superlative keyword is in the expression")
    parser.add_argument("--possessive_no_expand", action="store_true", help="(Parse) Expand ent2 in possessive case")
    args = parser.parse_args()

    # 加载{图片id : 文本描述}类型数据
    with open(args.input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    
    img_prer=IMAGE_PREPROCESS(image_path=args.image_root, input_file=args.input_file, detector_file_path=args.detector_file)
    img_dict=img_prer(out_path=args.img_cache_path, methods=args.box_representation_method.split(','), enlarge_boxes=args.enlarge_boxes, is_dynamic=args.dynamic_color_width, 
                      resize_square=args.resize_square, fine_grained=args.fine_grained, sam_cache_path=args.sam_cache_path, sam_temp_path=args.sam_temp_path, sam_iter=args.sam_iter, sam_method=0)
    del img_prer
    
    # random sample Q
    if args.size_Q == 0:
        corpus_Q = None
    else:
        # 每个图片只sample一个句子
        # sampled_sentences = ['this is a photo of ' + random.sample(datum['sentences'], 1)[0] for datum in data]
        # print("生成corpus_Q成功!")
        # corpus_Q = random.sample(sampled_sentences, args.size_Q) 
        corpus_Q = get_best_corpus(size_Q=args.size_Q, data=data)   
    
    # 默认是baseline, baseline已看完
    method = METHODS_MAP[args.method](args)
    
    correct_count = 0
    total_count = 0
    
    # 加载{图片id（int）: 检测框（list）}类型数据
    if args.detector_file:
        detector_file = open(args.detector_file)
        detections_list = json.load(detector_file)
        if isinstance(detections_list, dict):
            detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
        else:
            detections_map = defaultdict(list)
            for detection in detections_list:
                detections_map[detection["image_id"]].append(detection["box"])
                
    # 数据集分成n个part
    if args.part is not None:
        num_parts = int(args.part.split(",")[0])
        part = int(args.part.split(",")[1])
        data = data[int(len(data)*part/num_parts):int(len(data)*(part+1)/num_parts)]
    batch_pred_box = []

    # if args.size_text_prompt == 0:
    #     Set_images = None
    # else:
    #     # random sample image_names
    #     sampled_images = [(str(datum['image_id']), Box(x=detections_map[int(datum['image_id'])][0][0], y=detections_map[int(datum['image_id'])][0][1], w=detections_map[int(datum['image_id'])][0][2], h=detections_map[int(datum['image_id'])][0][3])) for datum in data]
    #     temp_set = set()
    #     unique_sampled_images = []
    #     for image_id, detection in sampled_images:
    #         if image_id not in temp_set:
    #             unique_sampled_images.append((image_id, detection))
    #             temp_set.add(image_id)
    #     print(unique_sampled_images[0])
    #     Set_images = random.sample(unique_sampled_images, args.size_text_prompt)
    #     print("生成Set_images成功!")
    
    # 作为一个参数传进env
    # clip_model 默认是两个视觉模型
    # box_representation_method(将检测框图像表示成单独一个图像的方式) 默认是'裁剪,模糊'
    # box_method_aggregator(组合检测框图像表示分数的方法) 默认是求和
    # square_size(是不是正方形)
    # blur_std_dev(高斯模糊的标准差)

    executor = ClipExecutor(clip_model=args.clip_model, box_representation_method=args.box_representation_method, method_aggregator=args.box_method_aggregator, enlarge_boxes=args.enlarge_boxes, device=device, square_size=not args.non_square_size, expand_position_embedding=args.expand_position_embedding, blur_std_dev=args.blur_std_dev, fine_grained=args.fine_grained, sam_cache_path=args.sam_cache_path, sam_iter= args.sam_iter, resize_square=args.resize_square,  cache_path=args.cache_path, img_dict=img_dict, corpus_Q = corpus_Q, text_prompt = args.text_prompt)
    
    # data是文本数据，每个图片可能对应多个文本描述
    for datum in tqdm(data):
        file_name = datum["file_name"]
        img_path = os.path.join(args.image_root, file_name)
        img = Image.open(img_path).convert('RGB')
        for sentence in datum["sentences"]:
            boxes = [Box(x=box[0], y=box[1], w=box[2], h=box[3]) for box in detections_map[int(datum["image_id"])]]
            if len(boxes) == 0:
                boxes = [Box(x=0, y=0, w=img.width, h=img.height)]
                
            # img: 一张大图片
            # boxes: 该img对应的所有检测框，a list of object(BOX)
            # executor: 能调用CLIP模型的object
            # image_name: str(img_id)
            env = Environment(img, boxes, executor, freeform_boxes = (args.mdetr is not None and not args.mdetr_given_bboxes), image_name = str(datum["image_id"]), file_name = file_name, is_dynamic = args.dynamic_color_width)
            
            # 把一句话中的单词打乱顺序
            if args.shuffle_words:
                words = sentence.lower().split()
                random.shuffle(words)
                result = method.execute(" ".join(words), env)
            else:
                result = method.execute(sentence.lower(), env)
            
            boxes = env.boxes
            pred_box = boxes[result["pred"]]
            batch_pred_box.append([pred_box.left, pred_box.top, pred_box.right, pred_box.bottom])

    with open(args.results_path, 'w') as f:
        json.dump(batch_pred_box, f)
