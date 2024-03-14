# import json
#
#
# def bio_to_dict(bio_file_path, output_file_path):
#     with open(bio_file_path, 'r', encoding='utf-8') as bio_file:
#         lines = bio_file.readlines()
#
#     data = []
#     text = []
#     labels = []
#     id_count = 7883
#     for line in lines:
#         if line.strip() == "" or line.isspace():  # 如果行为空或只包含空白字符
#             if text and labels:  # 如果text和labels非空，则添加一个新的数据样本
#                 id_str = "AT" + str(id_count).zfill(4)  # 生成id
#                 id_count += 1
#
#                 data.append({"id": id_str, "text": text, "labels": labels})
#
#                 text = []  # 重置
#                 labels = []  # 重置
#         else:
#             word, label = line.strip().split(' ')  # 使用空格来分割
#             text.append(word)
#             labels.append(label)
#
#     # 处理文件最后一个数据样本（如果它没有被空行结束）
#     if text and labels:
#         id_str = "AT" + str(id_count).zfill(4)  # 生成id
#         data.append({"id": id_str, "text": text, "labels": labels})
#
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         for item in data:
#             output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
#
#
# # 使用函数
# bio_to_dict("E:\\BERT-BILSTM-CRF\\data\\geo2\\ori_data\\train.txt", "data/geo2/ner_data/train.txt")
import os
import json

def bio_to_dict(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    id_count = 1
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".txt"):
            bio_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, filename)

            with open(bio_file_path, 'r', encoding='utf-8') as bio_file:
                lines = bio_file.readlines()

            data = []
            text = []
            labels = []

            for line in lines:
                if line.strip() == "" or line.isspace():
                    if text and labels:
                        id_str = "AT" + str(id_count).zfill(4)
                        id_count += 1

                        data.append({"id": id_str, "text": text, "labels": labels})
                        text = []
                        labels = []
                else:
                    word, label = line.strip().split(' ')
                    text.append(word)
                    labels.append(label)

            if text and labels:
                id_str = "AT" + str(id_count).zfill(4)
                data.append({"id": id_str, "text": text, "labels": labels})

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for item in data:
                    output_file.write(json.dumps(item, ensure_ascii=False) + '\n')

# 使用函数
input_folder_path = "E:\\BERT-BILSTM-CRF\\data\\geo\\ori_data"
output_folder_path = "E:\\BERT-BILSTM-CRF\\data\\geo\\ner_data"

bio_to_dict(input_folder_path, output_folder_path)
