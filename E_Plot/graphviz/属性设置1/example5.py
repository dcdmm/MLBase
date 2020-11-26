from graphviz import Digraph

d = Digraph(name='example5', filename='example5.dot')

with d.subgraph(name='cluster_0') as s:
    s.attr(style='bold', color='red')
    s.attr(rank='same') # 节点都位于相同的秩上
    s.node('A')
    s.node('X')

d.node('C')

with d.subgraph(name='cluster_2') as s:
    s.attr(rank='min') # 节点都位于最小的秩上(最顶部或最左侧)
    s.node('B')
    s.node('D')
    s.node('Y')

with d.subgraph() as s: # 节点都位于最大的秩上(最底部或最右侧)
    s.attr(rank='max')
    s.node('F')
    s.node('G')

d.edges(['AB', 'AC', 'CD', 'XY', 'BF', 'DG', 'AG'])

d.view()