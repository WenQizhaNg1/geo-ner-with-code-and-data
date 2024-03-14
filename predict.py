import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin")))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    data_name = "geo2"
    predictor = Predictor(data_name)
    if data_name == "geo3":
        texts = [
            "Simpat算法是首个基于数据样式相似度建模的算法,该方法从样式数据库里查询与数据事件相似度最高的数据样式,通过替换、冻结数据事件的所有节点完成对未知部分的预测。改进算法Filtersim基于过滤器对数据样式实现聚类,进一步提高建模效率。DisPat算法采用MDS降维和K均值聚类对样式数据库进行聚类,提高了建模效率;",
            "考虑到地下不同岩相储层参数存在明显差异,有学者提出利用混合高斯模型将岩相影响融合到随机反演中,提出了基于混合高斯模型的随机反演方法(GMMc)",
            "如分形域、马尔可夫域、模糊随机场、神经网络等。2.储层建模方法的应用研究储层建模方法及算法的发展使储层建模技术的应用领域和应用范围更为广泛,从油藏勘探阶段储层表征到开发中后期储层表征储层建模技术均有有效的应用",
        ]
    elif data_name == 'geo2':

        with open('candidate.txt','r',encoding='utf-8') as fp:
            texts = fp.read().split("\n")
        texts = [
            "其中古近系与下伏中生界地层呈角度不整合接触关系",
            "构成上亚段的是颜色呈灰色的部分，总体厚度大约为200m，顶部与沙三段接触，界限是盐底的页岩集中带，主要岩性是泥质深灰色泥岩、页岩、钙质页岩以及劣质油页岩的互层，也含有少量的泥质粉砂",
            "主要造岩矿物为钾长石50%、斜长石22.5%、石英27.5%、黑云母1.0%（局部可达3~5%）",
            "湖北大冶市杨文昌至许家湾白垩系下统大寺组二段（K1d2）－四段（K1d4）实测地层岩相剖面（PM017）（图2-41）未见顶大寺组四段（K1d4）56.浅灰紫色安山岩与安山质火山角砾岩互层。",
            "以变泥砂质岩中有细小鳞片状绢云母大量出现为特征。说明该变质带特征变质矿物以绿泥石—绢云母为主，变质程度较低。",
            "流纹岩：出露于阳春市山表村附近，岩石呈灰色块状，发育条带状流动构造，具斑状结构。",
            "变质矿物组合有：Pl+Bit±Mu+Qz±Alm，Chl±Ep+Qz±Cal，Pl+Kf+Bit+Qz，Hb(绿色)+Pl±Bit，Pl+Act+Chl±Ep，Bit+Mu+Qz±Gt。",
            "综上所述，浮土及碎屑岩类具低阻、低磁、低磁、高阻高密度特征；碳酸盐岩具低密度特性；火成岩除具中等磁性外，其他均介于上述二种岩性之间，但由酸性－基性电阻率、密度均有逐渐增高的174趋势（流纹岩电性及安山岩密度例外）",
            "厚34.89米30图2-24湖北省大冶市毛百市三叠系下-中统嘉陵江组实测地层剖面图2-23湖北省大冶市西畈李三叠系下统大冶组实测地层剖面3112.浅灰色块状含白云石微-细晶灰岩。",
            "油田构造演化经历了裂陷前期、裂陷早期、裂陷期、凹陷期、漂移期［９－１１］，其中裂陷前期对应于前寒武系结晶基底；裂陷早期系指距今１３０Ｍａ之前的地层，对应于纽康姆期岩浆活动形成的火成岩；",
            "四川盆地龙马潭凹陷西南中基性火山岩储层。",
            "在三角洲相中的火山岩相",
            "本组地层主要为含白云石微-细晶灰岩"
        ]
    result = []
    for text in texts:
        ner_result = predictor.ner_predict(text)
        result.append(ner_result)
        print("文本>>>>>：", text)
        print("实体>>>>>：", ner_result)
        print("="*100)

    # with open('result.json','w',encoding='utf-8') as fp:
    #     json.dump(result,fp,ensure_ascii=False)



