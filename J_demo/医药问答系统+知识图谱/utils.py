import ahocorasick
from transformers import AutoTokenizer, AutoModel
import torch
import os
import jieba
import re
import string

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


def editDistanceDP(w1, w2):
    """计算字符串之间的编辑距离"""
    m = len(w1)
    n = len(w2)
    solution = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(len(w2) + 1):
        solution[0][i] = i
    for i in range(len(w1) + 1):
        solution[i][0] = i

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i - 1] == w2[j - 1]:
                solution[i][j] = solution[i - 1][j - 1]
            else:
                solution[i][j] = 1 + min(solution[i][j - 1], min(solution[i - 1][j],
                                                                 solution[i - 1][j - 1]))
    return solution[m][n]


def simcal(question, stopwords, wdtype_dict, word2vec):
    # 数据清洗
    question = re.sub("[" + string.punctuation + "]", " ", question)
    question = re.sub("[，。‘’；：？、！【】]", " ", question)
    question = question.strip()

    # jieba中文分词,返回值为生成器
    words = [w.strip() for w in jieba.cut(question) if w.strip() not in stopwords and len(w.strip()) >= 2]

    result = {}
    for w in words:
        temp_result = []
        for key in wdtype_dict.keys():
            try:
                sim_score = word2vec.similarity(w, key)  # word2vec词相似度
            except KeyError:
                sim_score = 0
            dp_score = 1 - editDistanceDP(w, key) / (len(w) + len(key))  # 字符串编辑距离分数
            score_sum = 0.6 * sim_score + 0.4 * dp_score
            if score_sum > 0.5:
                temp_result.append((w, wdtype_dict.get(key), score_sum))
        if temp_result:
            temp_result.sort(key=lambda x: x[2], reverse=True)  # 选择最高分数代表的标签
        result.update({temp_result[0][0]: temp_result[0][1]})
    return result


if __name__ == '__main__':
    word1, word2 = 'dc', 'dmm'
    print(editDistanceDP(word1, word2))  # 测试通过
