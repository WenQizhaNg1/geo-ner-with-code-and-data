# -*- coding: utf-8 -*-
import os
import random
import jieba.posseg as psg
import jieba
import csv
import argparse

def add_entity(dict_dir):
    dict_path = os.path.join(os.getcwd(), dict_dir)
    with open(dict_path, 'r', encoding='utf8') as file:
        dics = csv.reader(file)
        for row in dics:
            if len(row) == 2:
                word, tag = row[0].strip(), row[1].strip()
                jieba.add_word(word, tag=tag)
                jieba.suggest_freq(word, tune=True)

def auto_label(input_texts, data_type, label_set, end_words=set(['。', '.', '?', '？', '!', '！']), mode='cn'):
    output_file = f"output_{mode}_{data_type}.txt"
    with open(output_file, "w", encoding="utf8") as writer:
        for input_text in input_texts:
            words = psg.cut(input_text)
            for word, pos in words:
                word, pos = word.strip(), pos.strip()
                if not (word and pos):
                    continue
                if mode == 'en':
                    word = word.split(' ')
                if pos not in label_set:
                    for char in word:
                        string = char + ' ' + 'O' + '\n'
                        if char in end_words:
                            string += '\n'
                        writer.write(string)
                else:
                    for i, char in enumerate(word):
                        bio_tag = 'B-' if i == 0 else 'I-'
                        string = char + ' ' + bio_tag + pos + '\n'
                        writer.write(string)

def load_data(source_data_dir):
    lines = []
    for file in os.listdir(source_data_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(source_data_dir, file)
            with open(file_path, 'r', encoding='utf8') as fp:
                lines.extend(line.strip() for line in fp if line.strip())
    return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare weak supervised data.')
    parser.add_argument('--language', type=str, default='cn')
    parser.add_argument('--source_dir', type=str, default='source_data')
    parser.add_argument('--dict_dir', type=str, default='dict.csv')
    parser.add_argument('--train_rate', type=float, default=0.8)
    parser.add_argument('--dev_rate', type=float, default=0.1)
    parser.add_argument('--test_rate', type=float, default=0.1)

    args = parser.parse_args()

    mode = args.language
    source_data_dir = os.path.join(os.getcwd(), args.source_dir)
    dict_dir = args.dict_dir
    train_rate, dev_rate, test_rate = args.train_rate, args.dev_rate, args.test_rate

    assert train_rate + dev_rate + test_rate == 1.0, "The sum of rate arguments must be 1."

    add_entity(dict_dir)

    with open(os.path.join(os.getcwd(), dict_dir), 'r', encoding='utf8') as file:
        dics = csv.reader(file)
        label_set = {row[1].strip() for row in dics if len(row) == 2}

    lines = load_data(source_data_dir)

    print(f'Preparing the weak supervised dataset for language: {mode} ...')
    print(f'The source data directory: {source_data_dir}')
    print(f'Total length of corpus: {len(lines)}')

    assert lines, "The source data directory is empty or no valid text files were found."

    random.seed(42)
    random.shuffle(lines)

    train_end_idx = int(train_rate * len(lines))
    dev_end_idx = int((train_rate + dev_rate) * len(lines))

    auto_label(lines[:train_end_idx], 'train', label_set, mode=mode)
    auto_label(lines[train_end_idx:dev_end_idx], 'dev', label_set, mode=mode)
    auto_label(lines[dev_end_idx:], 'test', label_set, mode=mode)

    print('Building success.')
