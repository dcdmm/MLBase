import ahocorasick
from transformers import AutoTokenizer, AutoModel
import torch
import os

from QIC.model import Model


def build_entitydict(args):
    # input:{'苯中毒': ['disease'], '肺炎': ['disease'], '黄酒': ['food']}
    # result:{'disease': ['苯中毒', '肺炎'], 'food': ['黄酒']}
    entity_dict = {}
    for arg, types in args.items():
        for ts in types:
            if ts not in entity_dict:
                entity_dict[ts] = [arg]
            else:
                entity_dict[ts].append(arg)
    return entity_dict


def build_actree(wordlist):
    actree = ahocorasick.Automaton()  # create an Automaton
    for index, word in enumerate(wordlist):
        actree.add_word(word, (index, word))
    actree.make_automaton()  # convert the trie to an Aho-Corasick automaton to enable Aho-Corasick search
    return actree


def cus_update(d1, d2):
    """自定义字典更新方式,用于替代字典内置update方法"""
    for key in d2.keys():
        if key in d1:
            d1[key].extend(d2[key])
        else:
            d1[key] = d2[key]


def check_words(wds, sent):  # wds中是否有词属于sent
    for wd in wds:
        if wd in sent:
            return True
    return False


def check_medical(region_tree, wdtype_dict, question):
    region_wds = []
    for i in region_tree.iter(question):  # 匹配所有字符串
        wd = i[1][1]
        region_wds.append(wd)
    stop_wds = []
    for wd1 in region_wds:
        for wd2 in region_wds:
            if wd1 in wd2 and wd1 != wd2:
                stop_wds.append(wd1)
    final_wds = [i for i in region_wds if i not in stop_wds]
    final_dict = {i: wdtype_dict.get(i) for i in final_wds}
    return final_dict


def model_intention_predict(question):
    """意图分类,通过KUAKE-QIC数据集训练得"""
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_dir, 'QIC/torch_model.bin')

    token = AutoTokenizer.from_pretrained("nghuyong/ernie-health-zh")
    best_model = Model(AutoModel.from_pretrained("nghuyong/ernie-health-zh"))
    best_model.load_state_dict(torch.load(model_path))  # 加载最优模型
    text_encode = token([question], return_tensors='pt')
    predict = best_model(text_encode['input_ids'], text_encode['attention_mask'], text_encode['token_type_ids'])
    if torch.max(predict) > 0.85:
        if torch.argmax(predict) == 0:  # 对应病情诊断
            return 'symptom_disease'
        if torch.argmax(predict) == 1:  # 对应病因分析
            return 'disease_cause'
        if torch.argmax(predict) == 2:  # 对应治疗方案
            return 'disease_cureway'
        if torch.argmax(predict) == 3:  # 对应就医建议
            return 'disease_check'
        if torch.argmax(predict) == 5:  # 对应疾病表述
            return 'symptom_qwds'
    return ''
