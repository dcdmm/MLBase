from graphviz import Digraph

g = Digraph(name='example2', filename='example2.dot')

# 创建子图与设置style属性
with g.subgraph(name='cluster_0') as c:  # the subgraph name needs to begin with 'cluster' (all lowercase)
    c.attr(style='filled', color='yellow')  # 子图属性设置
    c.attr('node', style='filled', color='green')  # 节点属性设置
    c.attr('edge', color='red')  # 边属性设置
    c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])  # Create a bunch of edges
    c.attr(label='process #1')

with g.subgraph(name='subgraph') as c:  # 若子图name属性不以cluster开头,则不能将图形包含在矩形边框内
    c.attr(style='bold', color='red')  # 设置无效
    c.attr('node', style='filled')
    c.edges([('b0', 'b1'), ('b1', 'b2'), ('b2', 'b3')])
    c.attr(label='process #2')  # 子图标题

g.node('start', shape='diamond')
g.node('end', shape='square')

g.edge('start', 'a0')
g.edge('start', 'b0')
g.edge('a1', 'b3')
g.edge('b2', 'a3')
g.edge('a3', 'a0')
g.edge('a3', 'end')
g.edge('b3', 'end')

g.view()
