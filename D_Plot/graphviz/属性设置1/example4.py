from graphviz import Digraph

g = Digraph(name='example4', filename='example4.dot')
g.attr(compound='true')

with g.subgraph(name='cluster0') as c:
    c.edges(['ab', 'ac', 'bd', 'cd'])

with g.subgraph(name='cluster1') as c:
    c.edges(['eg', 'ef'])

g.edge('b', 'cluster1') # 从b指向cluster1(必须设定compound='true')
g.edge('d', 'e')
g.edge('c', 'g', ltail='cluster0', lhead='cluster1') # 从cluster0指向cluster1(必须设定compound='true')
g.edge('c', 'e', ltail='cluster0') # 从cluster1指向e(必须设定compound='true')
g.edge('d', 'h')

g.view()