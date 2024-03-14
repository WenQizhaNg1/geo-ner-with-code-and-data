import argparse
import random
from collections import defaultdict
import jieba
import json
from tqdm import tqdm
import synonyms

# 读取数据并读取数据集的词典
def read_bio_dataset_and_build_entity_dict(file_path):
    entity_dict = defaultdict(set)  # 存储同种类的BIO词
    dataset = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        current_entity = ''
        current_entity_type = ''
        
        for line in f:
            if line.strip():  # 非空行
                word, tag = line.strip().split(' ')
                sentence.append((word, tag))
                # 收集实体词信息
                if tag.startswith('B-'):
                    if current_entity:  # 保存之前的实体
                        entity_dict[current_entity_type].add(current_entity)
                    
                    current_entity = word
                    current_entity_type = tag[2:]
                elif tag.startswith('I-'):
                    current_entity += word
                else:
                    if current_entity:  # 保存之前的实体
                        entity_dict[current_entity_type].add(current_entity)
                    current_entity = ''
                    current_entity_type = ''
                    
            else:  # 空行，句子结束
                dataset.append(sentence)
                sentence = []
    
    # 返回数据集和实体字典
    return dataset, entity_dict

# 同义词替换改进，随机选择前五个同义词之一
def synonym_replacement(sentence):
    words, labels = zip(*sentence)  # 解压句子中的单词和标签
    words = list(words)
    labels = list(labels)
    non_entity_indices = [i for i, label in enumerate(labels) if label == 'O']  # 找到非实体词的索引
    
    if non_entity_indices:
        idx = random.choice(non_entity_indices)  # 随机选择一个非实体词
        word = words[idx]
        # 使用synonyms.nearby找到最接近的同义词
        synonyms_list, _ = synonyms.nearby(word)
        synonyms_list = synonyms_list[:5]

        if synonyms_list:
            # 随机选择一个同义词进行替换
            chosen_synonym = random.choice(synonyms_list)
            words[idx] = chosen_synonym
        
    return list(zip(words, labels))


# 随机插入
# TODO 这里还是考虑一下要不要做随机插入，感觉NER任务做随机插入是没啥作用的
# TODO 实验证明，NER任务做随机插入相当于随机插入噪声，可以微弱泛化性能
def random_insertion(sentence):
    words, labels = zip(*sentence)
    words = list(words)
    labels = list(labels)
    non_entity_indices = [i for i, label in enumerate(labels) if label == 'O']
    
    if len(non_entity_indices) > 0:
        idx = random.choice(non_entity_indices)
        word = words[idx]
        synonyms = SYNONYMS.get(word, [])
        if len(synonyms) > 0:
            insertion_idx = random.choice(non_entity_indices)
            words.insert(insertion_idx, random.choice(synonyms))
            labels.insert(insertion_idx, 'O')
            
    return list(zip(words, labels))

# 随机交换 完成 
# TODO 优化随机交换的策略，更好地模拟论文中的噪声
def random_swap(sentence, swaps=1):
    if not sentence or swaps < 1:
        return sentence  # 如果句子为空或无需交换，则直接返回

    words, labels = zip(*sentence)
    words = list(words)
    labels = list(labels)
    non_entity_indices = [i for i, label in enumerate(labels) if label == 'O']
    
    num_swaps = min(swaps, len(non_entity_indices) // 2) 

    for _ in range(num_swaps):
        if len(non_entity_indices) > 1:
            idx1, idx2 = random.sample(non_entity_indices, 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            non_entity_indices.remove(idx1)
            non_entity_indices.remove(idx2)

    return list(zip(words, labels))


# 加载停用词表
def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords
    
# 随机删除 完成
def random_deletion(sentence_with_bio, num_deletions=1, stopwords_path=None):
    # 读取停用词
    stopwords = set()
    if stopwords_path:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
                
    # 准备句子和标签
    words, tags = zip(*sentence_with_bio)
    sentence = ''.join(words)
    
    # 使用jieba进行分词
    segmented_words = list(jieba.cut(sentence))
    
    # 找出可以删除的非实体词的索引
    deletable_indices = []
    idx = 0
    for word in segmented_words:
        length = len(word)
        word_tags = tags[idx: idx + length]
        
        if all(tag == 'O' for tag in word_tags) and word not in stopwords:
            deletable_indices.append(idx)
        
        idx += length
    
    # 随机删除
    if len(deletable_indices) >= num_deletions:
        indices_to_delete = random.sample(deletable_indices, num_deletions)
    else:
        indices_to_delete = deletable_indices
    
    # 删除词和更新BIO标签
    new_sentence_with_bio = []
    idx = 0
    for word in segmented_words:
        length = len(word)
        
        if idx not in indices_to_delete:
            new_sentence_with_bio.extend(sentence_with_bio[idx: idx + length])
        
        idx += length
    
    return new_sentence_with_bio


# 实体交换函数 完成
def swap_entities(sentence, entity_dict):
    swapped_sentence = []
    skip_count = 0  # 跳过已处理实体的'I-'标签计数器
    
    for i, (word, tag) in enumerate(sentence):
        if skip_count > 0:  # 跳过已处理实体的后续'I-'标签
            skip_count -= 1
            continue
        
        if tag.startswith('B-'):
            entity_type = tag[2:]
            
            # 找出完整实体
            j = i + 1
            full_entity = [word]
            while j < len(sentence) and sentence[j][1] == f'I-{entity_type}':
                full_entity.append(sentence[j][0])
                j += 1
            
            # 计算需要跳过的'I-'标签数量
            skip_count = j - i - 1
            
            # 随机选择一个要替换的实体
            if entity_type in entity_dict and entity_dict[entity_type]:
                replacement_entity = random.choice(list(entity_dict[entity_type]))  # 转为列表
            else:
                replacement_entity = "".join(full_entity)  # 如果字典中没有该类型，则不进行替换
            
            # 添加替换后的实体及其BIO标签到新句子中
            replacement_entity_list = list(replacement_entity)
            for idx, char in enumerate(replacement_entity_list):
                if idx == 0:
                    swapped_sentence.append((char, f'B-{entity_type}'))
                else:
                    swapped_sentence.append((char, f'I-{entity_type}'))
                    
        else:
            swapped_sentence.append((word, tag))
            
    return swapped_sentence


# 实体切分函数 完成
def split_entity(sentence):

    new_sentence = []
    i = 0
    while i < len(sentence):
        word, tag = sentence[i]
        
        if tag.startswith('B-'):
            entity_type = tag[2:]
            full_entity_words = [word]
            j = i + 1
            
            # 收集完整的实体
            while j < len(sentence) and sentence[j][1].startswith('I-'):
                if sentence[j][1] == f'I-{entity_type}':  # 确保是相同实体类型的继续部分
                    full_entity_words.append(sentence[j][0])
                j += 1
            
            # 将实体词汇合并为一个字符串，并尝试切分
            entity_str = "".join(full_entity_words)
            split_parts = list(jieba.cut(entity_str, cut_all=False))
            
            # 仅在可以成功切分实体时处理
            if len(split_parts) > 1:
                retained_part = split_parts[-1]
                retained_part_words = list(retained_part)
                
                # 更新句子，保留切分后的部分并更新BIO标记
                for idx, char in enumerate(retained_part_words):
                    if idx == 0:
                        new_sentence.append((char, f'B-{entity_type}'))
                    else:
                        new_sentence.append((char, f'I-{entity_type}'))
            else:
                # 实体无法切分，按原样保留
                new_sentence.extend([(char, f'B-{entity_type}' if idx == 0 else f'I-{entity_type}') for idx, char in enumerate(full_entity_words)])
                
            i = j  # 跳过当前处理的实体
        else:
            new_sentence.append((word, tag))
            i += 1
    return new_sentence

# 新增实体类型 完成
def enhance_entities_with_dict(sentence, entity_dict):
    
    # with open(entity_path, 'r', encoding='utf-8') as f:
    #     entity_dict = json.load(f)
    # print(entity_dict)
    enhanced_sentence = []
    skip_count = 0  # 跳过已处理实体的'I-'标签计数器
    
    for i, (word, tag) in enumerate(sentence):
        if skip_count > 0:  # 跳过已处理实体的后续'I-'标签
            skip_count -= 1
            continue
        
        if tag.startswith('B-'):
            entity_type = tag[2:]
            
            # 找出完整实体
            j = i + 1
            full_entity = [word]
            while j < len(sentence) and sentence[j][1] == f'I-{entity_type}':
                full_entity.append(sentence[j][0])
                j += 1
            
            # 计算需要跳过的'I-'标签数量
            skip_count = j - i - 1
            
            # 随机选择一个要替换的实体
            if entity_type in entity_dict and entity_dict[entity_type]:
                replacement_entity = random.choice(entity_dict[entity_type])  # 直接使用列表
            else:
                replacement_entity = "".join(full_entity)  # 如果字典中没有该类型，则不进行替换
            
            # 添加替换后的实体及其BIO标签到新句子中
            replacement_entity_list = list(replacement_entity)
            for idx, char in enumerate(replacement_entity_list):
                if idx == 0:
                    enhanced_sentence.append((char, f'B-{entity_type}'))
                else:
                    enhanced_sentence.append((char, f'I-{entity_type}'))
                    
        else:
            enhanced_sentence.append((word, tag))
            
    return enhanced_sentence

def has_entity(data):
    # 判断一个句子是否含有实体
    for _, tag in data:
        if tag.startswith('B-') or tag.startswith('I-'):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Read BIO-formatted Named Entity Recognition dataset.")
    
    parser.add_argument('--input',default='geo/train.txt', type=str, required=False, help="Input file path for BIO-formatted dataset.")
    parser.add_argument('--output',default='geo/aug_train.txt', type=str, required=False, help="Output file path to save processed data.")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # 读取数据集
    # dataset = read_bio_dataset(input_path)
    dataset, entity_dict = read_bio_dataset_and_build_entity_dict(input_path)
    
    with open('dict/combined_data.json', 'r', encoding='utf-8') as f:
        entity_dict = json.load(f)
    
    # print(dataset[3])
    # print(swap_entities(dataset[3],entity_dict=entity_dict))
    # print(random_deletion(dataset[3],2,'cn_stopwords.txt'))
    # print(enhance_entities_with_dict(dataset[3],entity_dict=entity_dict))
    

    aug_sentence = []
    for data in dataset:
        # 自己组合增强策略
        pass
    
    # 保存数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in aug_sentence:
            for word, tag in sentence:
                f.write(f"{word} {tag}\n")  # 用制表符分隔词和标签
            f.write("\n")  # 在每个句子后添加一个空行以分隔

if __name__ == "__main__":
    main()
