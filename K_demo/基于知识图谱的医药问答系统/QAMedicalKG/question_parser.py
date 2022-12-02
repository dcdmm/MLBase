class QuestionPaser:
    """根据问题生成对应的CQL查询语句"""

    def parser_main(self, res_classify):
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        question_types = res_classify['question_types']

        cqls = []
        for question_type in question_types:
            cql_ = {'question_type': question_type}

            if question_type == 'disease_symptom':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'symptom_disease':
                cql = self.cql_transfer(question_type, entity_dict.get('symptom'))
            elif question_type == 'disease_cause':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_acompany':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_not_food':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_do_food':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'food_not_disease':
                cql = self.cql_transfer(question_type, entity_dict.get('food'))
            elif question_type == 'food_do_disease':
                cql = self.cql_transfer(question_type, entity_dict.get('food'))
            elif question_type == 'disease_drug':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'drug_disease':
                cql = self.cql_transfer(question_type, entity_dict.get('drug'))
            elif question_type == 'disease_check':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'check_disease':
                cql = self.cql_transfer(question_type, entity_dict.get('check'))
            elif question_type == 'disease_prevent':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_lasttime':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_cureway':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_cureprob':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_easyget':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            elif question_type == 'disease_desc':
                cql = self.cql_transfer(question_type, entity_dict.get('disease'))
            else:
                cql = []

            if cql:
                cql_['cql'] = cql
                cqls.append(cql_)
        return cqls

    def build_entitydict(self, args):
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

    def cql_transfer(self, question_type, entities):
        if not entities:
            return []

        # neo4j查询疾病原因的CQL语句
        if question_type == 'disease_cause':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cause".format(i) for i in entities]
        # neo4j查询疾病的防御措施的CQL语句
        elif question_type == 'disease_prevent':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.prevent".format(i) for i in entities]
        # neo4j查询疾病的持续时间的CQL语句
        elif question_type == 'disease_lasttime':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cure_lasttime".format(i) for i in entities]
        # neo4j查询疾病的治愈概率的CQL语句
        elif question_type == 'disease_cureprob':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cured_prob".format(i) for i in entities]
        # neo4j查询疾病的治疗方式的CQL语句
        elif question_type == 'disease_cureway':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.cure_way".format(i) for i in entities]
        # neo4j查询疾病的易发人群的CQL语句
        elif question_type == 'disease_easyget':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.easy_get".format(i) for i in entities]
        # neo4j查询疾病的相关介绍的CQL语句
        elif question_type == 'disease_desc':
            cql = ["MATCH (m:diseases) where m.name = '{0}' return m.name, m.desc".format(i) for i in entities]
        # neo4j查询疾病有哪些症状的CQL语句
        elif question_type == 'disease_symptom':
            cql = [
                "MATCH (m:diseases)-[r:has_symptom]->(n:symptoms) where m.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
        # neo4j查询症状会导致哪些疾病的CQL语句
        elif question_type == 'symptom_disease':
            cql = [
                "MATCH (m:diseases)-[r:has_symptom]->(n:symptoms) where n.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
        # neo4j查询疾病的并发症的CQL语句
        elif question_type == 'disease_acompany':
            cql = [
                "MATCH (m:diseases)-[r:acompany_with]->(n:diseases) where m.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
        # neo4j查询疾病的忌口的CQL语句
        elif question_type == 'disease_not_food':
            cql = ["MATCH (m:diseases)-[r:no_eat]->(n:foods) where m.name = '{0}' return m.name, r.name, n.name".
                       format(i) for i in entities]
        # neo4j查询查询疾病建议吃的东西的CQL语句
        elif question_type == 'disease_do_food':
            cql1 = ["MATCH (m:diseases)-[r:do_eat]->(n:foods) where m.name = '{0}' return m.name, r.name, n.name".
                        format(i) for i in entities]
            cql2 = [
                "MATCH (m:diseases)-[r:recommand_eat]->(n:foods) where m.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
            cql = cql1 + cql2
        # neo4j已知忌口查疾病的CQL语句
        elif question_type == 'food_not_disease':
            cql = [
                "MATCH (m:diseases)-[r:no_eat]->(n:foods) where n.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
        # neo4j已知推荐查疾病的CQL语句
        elif question_type == 'food_do_disease':
            cql1 = [
                "MATCH (m:diseases)-[r:do_eat]->(n:foods) where n.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
            cql2 = [
                "MATCH (m:diseases)-[r:recommand_eat]->(n:foods) where n.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
            cql = cql1 + cql2
        # neo4j查询疾病常用药品
        elif question_type == 'disease_drug':
            cql1 = [
                "MATCH (m:diseases)-[r:common_drug]->(n:drugs) where m.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
            cql2 = [
                "MATCH (m:diseases)-[r:recommand_drug]->(n:drugs) where m.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
            cql = cql1 + cql2
        # neo4j已知药品查询能够治疗的疾病的CQL语句
        elif question_type == 'drug_disease':
            cql1 = ["MATCH (m:diseases)-[r:common_drug]->(n:drugs) where n.name = '{0}' return m.name, r.name, n.name".
                        format(i) for i in entities]
            cql2 = [
                "MATCH (m:diseases)-[r:recommand_drug]->(n:drugs) where n.name = '{0}' return m.name, r.name, n.name".
                    format(i) for i in entities]
            cql = cql1 + cql2
        # neo4j查询疾病应该进行的检查的CQL语句
        elif question_type == 'disease_check':
            cql = ["MATCH (m:diseases)-[r:need_check]->(n:checks) where m.name = '{0}' return m.name, r.name, n.name".
                       format(i) for i in entities]
        # neo4j已知检查查询疾病的CQL语句
        elif question_type == 'check_disease':
            cql = ["MATCH (m:diseases)-[r:need_check]->(n:checks) where n.name = '{0}' return m.name, r.name, n.name".
                       format(i) for i in entities]
        else:
            cql = []
        return cql  # CQL查询语句


if __name__ == '__main__':
    # 测试通过
    rc = {'args': {'苯中毒': ['disease'], '肺炎': ['disease']}, 'question_types': ['disease_do_food', 'disease_prevent']}
    qp = QuestionPaser()
    rc_sql = qp.parser_main(rc)
    '''
    [{'question_type': 'disease_do_food',
      'cql': ["MATCH (m:diseases)-[r:do_eat]->(n:foods) where m.name = '苯中毒' return m.name, r.name, n.name",
              "MATCH (m:diseases)-[r:do_eat]->(n:foods) where m.name = '肺炎' return m.name, r.name, n.name",
              "MATCH (m:diseases)-[r:recommand_eat]->(n:foods) where m.name = '苯中毒' return m.name, r.name, n.name",
              "MATCH (m:diseases)-[r:recommand_eat]->(n:foods) where m.name = '肺炎' return m.name, r.name, n.name"]},
     {'question_type': 'disease_prevent',
      'cql': ["MATCH (m:diseases) where m.name = '苯中毒' return m.name, m.prevent",
              "MATCH (m:diseases) where m.name = '肺炎' return m.name, m.prevent"]}]
    '''
    print(rc_sql)
