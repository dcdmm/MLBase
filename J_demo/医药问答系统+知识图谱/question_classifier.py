import os

from utils import cus_update, build_actree, check_words, check_medical, model_intention_predict, simcal


class QuestionClassifier:
    """问题分类"""

    def __init__(self):
        cur_dir = os.path.abspath(os.path.dirname(__file__))

        # 导出Node数据文件路径
        self.disease_path = os.path.join(cur_dir, 'nodes/diseases.txt')
        self.department_path = os.path.join(cur_dir, 'nodes/departments.txt')
        self.check_path = os.path.join(cur_dir, 'nodes/checks.txt')
        self.drug_path = os.path.join(cur_dir, 'nodes/drugs.txt')
        self.food_path = os.path.join(cur_dir, 'nodes/foods.txt')
        self.producer_path = os.path.join(cur_dir, 'nodes/producers.txt')
        self.symptom_path = os.path.join(cur_dir, 'nodes/symptoms.txt')

        self.wdtype_dict = {}
        # 一个实体可能对应多种标签,故实体标签使用列表存储
        cus_update(self.wdtype_dict,
                   {i.strip(): ['disease'] for i in open(self.disease_path, encoding="utf-8") if i.strip()})
        cus_update(self.wdtype_dict,
                   {i.strip(): ['department'] for i in open(self.department_path, encoding="utf-8") if i.strip()})
        cus_update(self.wdtype_dict,
                   {i.strip(): ['check'] for i in open(self.check_path, encoding="utf-8") if i.strip()})
        cus_update(self.wdtype_dict,
                   {i.strip(): ['drug'] for i in open(self.drug_path, encoding="utf-8") if i.strip()})
        cus_update(self.wdtype_dict,
                   {i.strip(): ['food'] for i in open(self.food_path, encoding="utf-8") if i.strip()})
        cus_update(self.wdtype_dict,
                   {i.strip(): ['producer'] for i in open(self.producer_path, encoding="utf-8") if i.strip()})
        cus_update(self.wdtype_dict,
                   {i.strip(): ['producer'] for i in open(self.symptom_path, encoding="utf-8") if i.strip()})
        self.region_tree = build_actree(list(self.wdtype_dict.keys()))

        # 否定词
        self.deny_path = os.path.join(cur_dir, 'extra_data/deny.txt')
        self.deny_words = [i.strip() for i in open(self.deny_path, encoding="utf-8") if i.strip()]

        # 关键字匹配(规则)
        self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']
        self.cause_qwds = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何', '如何才会',
                           '怎么才会', '会导致', '会造成']
        self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现',
                              '伴随发生',
                              '伴随', '共现']
        self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱', '菜谱',
                          '食用',
                          '食物', '补品']
        self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
        self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得', '逃开', '避开', '避掉',
                             '躲开', '躲掉', '绕开', '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不', '如何才能不',
                             '怎样才不',
                             '怎么才不', '咋样才不', '咋才不', '如何才不', '怎样才可以不', '怎么才可以不', '咋样才可以不',
                             '咋才可以不',
                             '如何可以不', '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时', '多少年']
        self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治',
                             '怎么办',
                             '咋办', '咋治']
        self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性', '能治',
                              '可治',
                              '可以治', '可以医']
        self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']
        self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']
        self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']
        self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用', '用处',
                          '用途', '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']

        self.model_path = os.path.join(cur_dir, 'QIC/torch_model.bin')

        self.stopwords = [w.strip() for w in
                          open(os.path.join(cur_dir, 'extra_data/stopwords.dat'), 'r', encoding='utf8') if w.strip()]

    def classify(self, question):
        data = {}

        medical_dict = check_medical(self.region_tree, self.wdtype_dict, question)
        if not medical_dict:
            medical_dict = simcal(question, self.stopwords, self.wdtype_dict)
            if not medical_dict:
                return {}
        data['args'] = medical_dict

        types = []  # 问句当中所涉及到的实体类型
        for type_ in medical_dict.values():
            types += type_
        question_types = []

        intention_predict = model_intention_predict(question)
        if intention_predict != '':
            question_types.append(intention_predict)

        if check_words(self.symptom_qwds, question) and ('disease' in types):
            question_type = 'disease_symptom'  # 疾病->症状
            question_types.append(question_type)

        if check_words(self.symptom_qwds, question) and ('symptom' in types):
            question_type = 'symptom_disease'  # 症状->疾病
            question_types.append(question_type)

        if check_words(self.cause_qwds, question) and ('disease' in types):
            question_type = 'disease_cause'  # 疾病->造成
            question_types.append(question_type)

        if check_words(self.acompany_qwds, question) and ('disease' in types):
            question_type = 'disease_acompany'  # 疾病->并发
            question_types.append(question_type)

        if check_words(self.food_qwds, question) and 'disease' in types:
            deny_status = check_words(self.deny_words, question)
            if deny_status:
                question_type = 'disease_not_food'  # 疾病->忌吃食物
            else:
                question_type = 'disease_do_food'  # 疾病->宜吃食物
            question_types.append(question_type)

        if check_words(self.food_qwds + self.cure_qwds, question) and 'food' in types:
            deny_status = check_words(self.deny_words, question)
            if deny_status:
                question_type = 'food_not_disease'  # 忌吃食物->疾病
            else:
                question_type = 'food_do_disease'  # 宜吃食物->疾病
            question_types.append(question_type)

        if check_words(self.drug_qwds, question) and 'disease' in types:
            question_type = 'disease_drug'  # 疾病->药物
            question_types.append(question_type)

        if check_words(self.cure_qwds, question) and 'drug' in types:
            question_type = 'drug_disease'  # 药物->疾病
            question_types.append(question_type)

        if check_words(self.check_qwds, question) and 'disease' in types:
            question_type = 'disease_check'  # 疾病->检查
            question_types.append(question_type)

        if check_words(self.check_qwds + self.cure_qwds, question) and 'check' in types:
            question_type = 'check_disease'  # 检查->疾病
            question_types.append(question_type)

        if check_words(self.prevent_qwds, question) and 'disease' in types:
            question_type = 'disease_prevent'  # 疾病->防范
            question_types.append(question_type)

        if check_words(self.lasttime_qwds, question) and 'disease' in types:
            question_type = 'disease_lasttime'  # 疾病->治疗周期
            question_types.append(question_type)

        if check_words(self.cureway_qwds, question) and 'disease' in types:
            question_type = 'disease_cureway'  # 疾病->治疗方式
            question_types.append(question_type)

        if check_words(self.cureprob_qwds, question) and 'disease' in types:
            question_type = 'disease_cureprob'  # 疾病->治愈可能性
            question_types.append(question_type)

        if check_words(self.easyget_qwds, question) and 'disease' in types:
            question_type = 'disease_easyget'  # 疾病->易感染人群
            question_types.append(question_type)

        # 若没有查到相关的外部查询信息,那么则将该疾病的描述信息返回
        if question_types == [] and 'disease' in types:
            question_types = ['disease_desc']

        # 若没有查到相关的外部查询信息,且该疾病的描述信息为空,那么则将该疾病的描述信息返回
        if question_types == [] and 'symptom' in types:
            question_types = ['symptom_disease']

        data['question_types'] = list(set(question_types))
        return data


if __name__ == '__main__':
    # input:苯中毒该吃什么
    # result:{'args': {'苯中毒': ['disease']}, 'question_types': ['disease_do_food']}

    # input:苯中毒和肺炎该吃什么
    # result:{'args': {'苯中毒': ['disease'], '肺炎': ['disease']}, 'question_types': ['disease_do_food']}

    # input:苯中毒和肺炎该吃什么该如何预防
    # result:{'args': {'苯中毒': ['disease'], '肺炎': ['disease']}, 'question_types': ['disease_do_food', 'disease_prevent']}

    # 测试通过
    q = QuestionClassifier()
    problem = input('input an problem:')
    result = q.classify(problem)
    print(result)
