from py2neo import Graph  # TODO ★★★★py2neo已停止更新维护,请替换为neo4j库


class AnswerSearcher:
    """CQL查询语句neo4j搜索并输出对应的答案"""

    def __init__(self):
        self.g = Graph('http://localhost:7474', auth=('neo4j', '123456'), name='medicalkg')
        self.num_limit = 5  # 最大查询结果数量限制

    def search_main(self, cqls):
        final_answers = []
        for cql_ in cqls:
            question_type = cql_['question_type']
            queries = cql_['cql']
            answers = []
            for query in queries:
                ress = self.g.run(query).data()  # 返回值类型为列表,列表元素类型为字典,字典键名为CQL返回值变量名
                answers.append(ress)
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_prettify(self, question_type, answers):
        final_answer = ''
        if not answers:
            return ''
        if question_type == 'disease_symptom':
            # "MATCH (m:diseases)-[r:has_symptom]->(n:symptoms) where m.name = '{0}' return m.name, r.name, n.name".
            for i in answers:  # 可能有多个疾病
                subject = i[0]['m.name']  # 疾病名称
                desc = [j['n.name'] for j in i]  # 症状名称
                final_answer += '{0}的症状包括：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'symptom_disease':
            for i in answers:
                subject = i[0]['n.name']
                desc = [j['m.name'] for j in i]
                final_answer += '症状{0}可能染上的疾病有：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_cause':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['m.cause'] for j in i]
                final_answer += '{0}可能的成因有：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_prevent':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['m.prevent'] for j in i]
                final_answer += '{0}的预防措施包括：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_lasttime':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['m.cure_lasttime'] for j in i]
                final_answer += '{0}治疗可能持续的周期为：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_cureway':
            for i in answers:
                subject = i[0]['m.name']
                desc = [';'.join(j['m.cure_way']) for j in i]
                final_answer += '{0}可以尝试如下治疗：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_cureprob':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['m.cured_prob'] for j in i]
                final_answer += '{0}治愈的概率为（仅供参考）：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_easyget':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['m.easy_get'] for j in i]
                final_answer += '{0}的易感人群包括：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_desc':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['m.desc'] for j in i]
                final_answer += '{0},熟悉一下：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_acompany':
            for i in answers:
                subject = i[0]['m.name']
                desc1 = [j['n.name'] for j in i]
                desc2 = [j['m.name'] for j in i]
                desc = [k for k in desc1 + desc2 if k != subject]
                final_answer += '{0}的症状包括：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_not_food':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['n.name'] for j in i]
                final_answer += '{0}忌食的食物包括有：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_do_food':
            for i in answers:
                subject = i[0]['m.name']
                if i[0]['r.name'] == '宜吃':
                    do_desc = [j['n.name'] for j in i]
                    answers = '{0}宜食的食物包括有：{1}。\n'.format(subject, ';'.join(list(set(do_desc))[:self.num_limit]))
                    final_answer += answers
                if i[0]['r.name'] == '推荐食谱':
                    recommand_desc = [j['n.name'] for j in i]
                    answers = '{0}推荐食谱的食物包括有：{1}。\n'.format(subject,
                                                            ';'.join(list(set(recommand_desc))[:self.num_limit]))
                    final_answer += answers
        elif question_type == 'food_not_disease':
            for i in answers:
                subject = i[0]['n.name']
                desc = [j['m.name'] for j in i]
                final_answer += '患有{0}的人最好不要吃{1}。\n'.format('；'.join(list(set(desc))[:self.num_limit]), subject)
        elif question_type == 'food_do_disease':
            for i in answers:
                subject = i[0]['n.name']
                desc = [j['m.name'] for j in i]
                final_answer += '患有{0}的人建议多试试{1}。\n'.format('；'.join(list(set(desc))[:self.num_limit]), subject)
        elif question_type == 'disease_drug':
            for i in answers:
                subject = i[0]['m.name']
                desc = [j['n.name'] for j in i]
                final_answer = '{0}通常的使用的药品包括：{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'drug_disease':
            for i in answers:
                subject = i[0]['n.name']
                desc = [j['m.name'] for j in i]
                final_answer = '{0}主治的疾病有{1},可以试试。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        elif question_type == 'disease_check':
            for i in answers:
                subject = i[0]['m.name']
                if i[0]['r.name'] == '诊断检查':
                    check_desc = [j['n.name'] for j in i]
                    final_answer += '{0}通常可以通过以下方式检查出来：{1}。\n'.format(subject,
                                                                      '；'.join(list(set(check_desc))[:self.num_limit]))
                if i[0]['r.name'] == '所属科室':
                    belong_to_desc = [j['n.name'] for j in i]
                    final_answer += '{0}的检查科室为：{1}。\n'.format(subject,
                                                              '；'.join(list(set(belong_to_desc))[:self.num_limit]))
        elif question_type == 'check_disease':
            for i in answers:
                subject = i[0]['n.name']
                desc = [j['m.name'] for j in i]
                final_answer += '通常可以通过{0}检查出来的疾病有{1}。\n'.format(subject, '；'.join(list(set(desc))[:self.num_limit]))
        if final_answer == '':
            return ''
        else:
            return final_answer[:-1]  # 去除字符串最后一个换行符
