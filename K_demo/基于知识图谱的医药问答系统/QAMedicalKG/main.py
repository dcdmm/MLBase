from question_classifier import *
from question_parser import *
from answer_search import *


class ChatBotGraph:
    """问答类"""

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = '没能理解您的问题，我数据量有限。。。能不能问的标准点'
        res_classify = self.classifier.classify(sent)  # 问题分类
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)  # 根据问题生成对应的CQL查询语句
        final_answers = self.searcher.search_main(res_sql)  # CQL查询语句neo4j搜索并输出对应的答案
        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)


if __name__ == '__main__':
    # 测试通过
    handler = ChatBotGraph()
    question = input('咨询:')
    problem = handler.chat_main(question)
    print('客服机器人:', problem)
