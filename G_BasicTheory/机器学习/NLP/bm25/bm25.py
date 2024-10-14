import numpy as np
import math

'''
参考:
https://github.com/dorianbrown/rank_bm25
https://www.cnblogs.com/geeks-reign/p/Okapi_BM25.html
'''


class BM25:
    """BM25类算法基类"""

    def __init__(self, corpus):
        self.corpus_size = 0  # 文档句子数
        self.avgdl = 0  # 文档平均每个句子的长度
        self.doc_freqs = []  # 文档每个句子中单词出现频率
        self.idf = {}
        self.doc_len = []  # 文档每个句子的长度

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # 整个文档中单词出现频率
        num_doc = 0  # 所有句子总长度(即文档总长度)
        '''
        corpus example:
        [['Hello', 'there', 'good', 'man!'],
         ['It', 'is', 'quite', 'windy', 'in', 'London'],
         ['How', 'is', 'the', 'weather', 'today?']]
        '''
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}  # 句子中单词出现频率
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1  # 文档句子数 + 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


# 参考:GitHubProjects\PyDevelopment\操作分布式搜索和分析引擎Elasticsearch\索引\setting参数_similarity参数.ipynb
class BM25Okapi(BM25):
    def __init__(self, corpus,
                 # 增加k1,词频较高的文档评分越高
                 # 减少k1,减弱词频对评分的影响(k1=0时,词频对评分无影响)
                 k1=1.5,
                 # 增加b,较长的文档收到更多的惩罚
                 # 减少b,减弱文档长度对评分的影响(b=0,文档长度对评分无影响)
                 b=0.75,
                 epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            # 见PDF 公式(4)
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            # 见PDF 公式(3)
            score += (self.idf.get(q) or 0) * (
                    q_freq * (self.k1 + 1) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        # L_d / L_{avg}偏好长度较短的句子
        self.delta = delta  # 避免算法对过长文本的惩罚
        super().__init__(corpus)

    def _calc_idf(self, nd):
        # 将不会取到负值
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            # 见PDF 公式(8)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
        self.k1 = k1
        self.b = b
        # That is, no matter how long the document, a single occurrence of a search term contributes at least a constant amount to the retrieval status value.
        self.delta = delta  # 作用类似BM25L
        super().__init__(corpus)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            # 见PDF 公式(9)
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) / (
                    self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()
