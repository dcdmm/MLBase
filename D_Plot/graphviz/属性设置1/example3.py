from graphviz import Graph

g = Graph(name='example3', filename='example3.dot')

# 边属性设置

g.attr(splines='ortho')
# splines='none'      无边
# splines='true'      默认
# splines='line'      直线
# splines='polyline'  折线
# splines='ortho'     直角折线
g.attr('edge', color='red') # 边的颜色
g.edge('run', 'intr')
g.edge('intr', 'runbl')
g.edge('runbl', 'run')
g.edge('run', 'kernel')
g.attr('edge', dir='both') # 箭头的边缘类型
g.edge('kernel', 'zombie')
g.edge('kernel', 'sleep')
g.edge('kernel', 'runmem')
g.edge('sleep', 'swap')
g.edge('swap', 'runswap')
g.edge('runswap', 'new')
g.edge('runswap', 'runmem')
g.edge('new', 'runmem')
g.edge('sleep', 'runmem')

g.view()