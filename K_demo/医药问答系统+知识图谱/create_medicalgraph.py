import os
import json
from py2neo import Graph, Node  # py2neo已停止更新维护,请替换为neo4j库


class MedicalGraph:
    """创建图数据库"""

    def __init__(self):
        self.data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'datasets/medical.json')
        self.g = Graph('http://localhost:7474', auth=('neo4j', '123456'), name='medicalkg')
        self.g.delete_all()

    def extract_info(self):
        # Node
        nodes = {'drugs': list(),  # 药品
                 'foods': list(),  # 食物
                 'checks': list(),  # 检查
                 'departments': list(),  # 科室
                 'producers': list(),  # 药品大类
                 'symptoms': list(),  # 症状
                 'diseases': list()}  # 疾病

        disease_infos = []  # 中心疾病信息
        diseases_acompany = list()  # 并发症疾病(且中心疾病不包含)

        # Relationship
        relations = {
            "rels_symptom": [list(), 'diseases', 'symptoms', 'has_symptom', '症状'],  # 疾病->症状
            "rels_acompany": [list(), 'diseases', 'diseases', 'acompany_with', '并发症'],  # 疾病->并发症
            "rels_department": [list(), 'departments', 'departments', 'belongs_to', '属于'],  # 科室->科室
            "rels_category": [list(), 'diseases', 'departments', 'belongs_to', '所属科室'],  # 疾病->科室
            "rels_commonddrug": [list(), 'diseases', 'drugs', 'common_drug', '常用药品'],  # 疾病->通用药品
            "rels_recommanddrug": [list(), 'diseases', 'drugs', 'recommand_drug', '好评药品'],  # 疾病->热门药品
            "rels_check": [list(), 'diseases', 'checks', 'need_check', '诊断检查'],  # 疾病->检查
            "rels_drug_producer": [list(), 'producers', 'drugs', 'drugs_of', '生产药品'],  # 厂商->药物
            "rels_recommandeat": [list(), 'diseases', 'foods', 'recommand_eat', '推荐食谱'],  # 疾病->推荐吃食物
            "rels_noteat": [list(), 'diseases', 'foods', 'no_eat', '忌吃'],  # 疾病->忌吃食物
            "rels_doeat": [list(), 'diseases', 'foods', 'do_eat', '宜吃']}  # 疾病->宜吃食物

        for data in open(self.data_path, encoding='utf-8'):
            data_json = json.loads(data)

            disease = data_json['name']
            nodes['diseases'].append(disease)
            disease_infos.append({'name': disease,
                                  'desc': data_json.setdefault('desc', ''),
                                  'prevent': data_json.setdefault('prevent', ''),
                                  'cause': data_json.setdefault('cause', ''),
                                  'get_prob': data_json.setdefault('get_prob', ''),
                                  'easy_get': data_json.setdefault('easy_get', ''),
                                  'cure_way': data_json.setdefault('cure_way', ''),
                                  'cure_lasttime': data_json.setdefault('cure_lasttime', ''),
                                  'cured_prob': data_json.setdefault('cured_prob', '')})

            if 'acompany' in data_json:
                acompany = data_json['acompany']
                relations['rels_acompany'][0].extend([(disease, i) for i in acompany])
                diseases_acompany.extend(acompany)

            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                    relations['rels_category'][0].append((disease, cure_department[0]))
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    relations['rels_department'][0].append((small, big))
                    relations['rels_category'][0].append((disease, small))
                nodes['departments'].extend(cure_department)

            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                relations['rels_commonddrug'][0].extend([(disease, i) for i in common_drug])
                nodes['drugs'].extend(common_drug)

            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                relations['rels_recommanddrug'][0].extend([(disease, i) for i in recommand_drug])
                nodes['drugs'].extend(recommand_drug)

            if 'check' in data_json:
                check = data_json['check']
                relations['rels_check'][0].extend([(disease, i) for i in check])
                nodes['checks'].extend(check)

            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                producer = [i.split('(')[0] for i in drug_detail]
                relations['rels_drug_producer'][0].extend(
                    [(i.split('(')[0], i.split('(')[-1].replace(')', '')) for i in drug_detail])
                nodes['producers'].extend(producer)

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                relations['rels_noteat'][0].extend([(disease, i) for i in not_eat])
                nodes['foods'].extend(not_eat)
            if 'do_eat' in data_json:
                do_eat = data_json['do_eat']
                relations['rels_doeat'][0].extend([(disease, i) for i in do_eat])
                nodes['foods'].extend(do_eat)
            if 'recommand_eat' in data_json:
                recommand_eat = data_json['recommand_eat']
                relations['rels_recommandeat'][0].extend([(disease, i) for i in recommand_eat])
                nodes['foods'].extend(recommand_eat)

            if 'symptom' in data_json:
                symptom = data_json['symptom']
                relations['rels_symptom'][0].extend([(disease, i) for i in symptom])
                nodes['symptoms'].extend(symptom)

        diseases_acompany = set(diseases_acompany)
        nodes = {k: set(v) for k, v in nodes.items()}
        relations = {k: [set(v[0]), v[1], v[2], v[3], v[4]] for k, v in relations.items()}

        diseases_acompany = diseases_acompany.difference(nodes['diseases'])  # 去除中心疾病中包含的疾病
        nodes['diseases'] = nodes['diseases'].union(diseases_acompany)
        return nodes, relations, disease_infos, diseases_acompany

    def create_node(self, label, nodes):
        for nodel_info in nodes:
            if isinstance(nodel_info, dict):
                node = Node(label, **nodel_info)
                self.g.create(node)
            else:
                node = Node(label, name=nodel_info)
                self.g.create(node)

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        for edge in edges:
            p = edge[0]
            q = edge[1]
            # CQL语法
            query = """match(p:%s),(q:%s) where p.name="%s"and q.name="%s" create (p)-[rel:%s{name:"%s"}]->(q)""" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
            except Exception as e:
                print(e)

    def create_graph(self, export_data=False):
        nodes, relations, disease_infos, diseases_acompany = self.extract_info()
        for key in nodes.keys():
            if export_data:
                # 导出Node数据
                with open('nodes/{}.txt'.format(key), 'w+', encoding='UTF-8') as f:
                    f.write('\n'.join(list(nodes[key])))
            if key != 'diseases':
                self.create_node(key, nodes[key])
        self.create_node('diseases', disease_infos)
        self.create_node('diseases', diseases_acompany)

        for key in relations.keys():
            self.create_relationship(relations[key][1], relations[key][2],
                                     relations[key][0],
                                     relations[key][3], relations[key][4])


if __name__ == '__main__':
    # 测试通过
    handler = MedicalGraph()
    handler.create_graph(export_data=True)
    print('over!')
