from py2neo import Graph


class AnswerSearcher:
    """CQL查询语句neo4j搜索并输出对应的答案"""

    def __init__(self):
        self.g = Graph('http://localhost:7474', auth=('neo4j', '123456'), name='test')
        self.num_limit = 20  # 查询结果数量限制

    def search_main(self, cqls):
        final_answers = []
        for cql_ in cqls:
            question_type = cql_['question_type']
            queries = cql_['cql']
            answers = []
            for query in queries:
                ress = self.g.run(query).data()  # 返回值类型为列表,列表元素类型为字典,字典键名为CQL返回值变量名
                answers += ress
                print('xxxxx', ress)
            print('xzzzzzzzzzz', answers)
            final_answer = self.answer_prettify(question_type, answers)
            print('yyyyy', final_answer)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_prettify(self, question_type, answers):
        final_answer = []
        if not answers:
            return ''
        if question_type == 'disease_symptom':
            # "MATCH (m:diseases)-[r:has_symptom]->(n:symptoms) where m.name = '{0}' return m.name, r.name, n.name".
            subject = answers[0]['m.name']  # 疾病名称(这里仅支持:出现实体数量为1)
            desc = [i['n.name'] for i in answers]  # 症状名称
            final_answer = '{0}的症状包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'symptom_disease':
            subject = answers[0]['n.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.name'] for i in answers]
            final_answer = '症状{0}可能染上的疾病有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_cause':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.cause'] for i in answers]
            final_answer = '{0}可能的成因有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_prevent':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.prevent'] for i in answers]
            final_answer = '{0}的预防措施包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_lasttime':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.cure_lasttime'] for i in answers]
            final_answer = '{0}治疗可能持续的周期为：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_cureway':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [';'.join(i['m.cure_way']) for i in answers]
            final_answer = '{0}可以尝试如下治疗：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_cureprob':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.cured_prob'] for i in answers]
            final_answer = '{0}治愈的概率为（仅供参考）：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_easyget':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.easy_get'] for i in answers]
            final_answer = '{0}的易感人群包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_desc':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.desc'] for i in answers]
            final_answer = '{0},熟悉一下：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_acompany':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc1 = [i['n.name'] for i in answers]
            desc2 = [i['m.name'] for i in answers]
            desc = [i for i in desc1 + desc2 if i != subject]
            final_answer = '{0}的症状包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_not_food':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['n.name'] for i in answers]
            final_answer = '{0}忌食的食物包括有：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_do_food':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            do_desc = [i['n.name'] for i in answers if i['r.name'] == '宜吃']
            recommand_desc = [i['n.name'] for i in answers if i['r.name'] == '推荐食谱']
            final_answer = '{0}宜食的食物包括有：{1}\n推荐食谱包括有：{2}'.format(subject, ';'.join(list(set(do_desc))[:self.num_limit]),
                                                                 ';'.join(list(set(recommand_desc))[:self.num_limit]))
        elif question_type == 'food_not_disease':
            subject = answers[0]['n.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.name'] for i in answers]
            final_answer = '患有{0}的人最好不要吃{1}'.format('；'.join(list(set(desc))[:self.num_limit]), subject)
        elif question_type == 'food_do_disease':
            subject = answers[0]['n.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.name'] for i in answers]
            final_answer = '患有{0}的人建议多试试{1}'.format('；'.join(list(set(desc))[:self.num_limit]), subject)
        elif question_type == 'disease_drug':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['n.name'] for i in answers]
            final_answer = '{0}通常的使用的药品包括：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'drug_disease':
            subject = answers[0]['n.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.name'] for i in answers]
            final_answer = '{0}主治的疾病有{1},可以试试'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_check':
            subject = answers[0]['m.name']  # 这里仅支持:出现实体数量为1
            desc = [i['n.name'] for i in answers]
            final_answer = '{0}通常可以通过以下方式检查出来：{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'check_disease':
            subject = answers[0]['n.name']  # 这里仅支持:出现实体数量为1
            desc = [i['m.name'] for i in answers]
            final_answer = '通常可以通过{0}检查出来的疾病有{1}'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        return final_answer
