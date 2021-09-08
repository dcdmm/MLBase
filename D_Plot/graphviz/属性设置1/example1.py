from graphviz import Digraph

f = Digraph('example1', filename='example1.dot', format='png')

# "TB", "逻辑斯蒂回归LR", "BT", "RL"
f.attr(rankdir='逻辑斯蒂回归LR')  # LR表示从左至右进行绘图(对整个图而言)

f.attr('node', shape='doublecircle')
f.node('LR_0', label='跟结点', fontname='SimHei')
f.node('LR_3')
f.node('LR_4')
f.node('LR_8')

f.attr('node', shape='circle')
f.node('LR_2')
f.node('LR_1')
f.node('LR_6')
f.node('LR_5')
f.node('LR_7')
f.node('LR_8')

f.edge('LR_5', 'LR_5', label='S(a)')
f.edge('LR_6', 'LR_6', label='S(b)')
f.edge('LR_6', 'LR_5', label='S(a)')
f.edge('LR_5', 'LR_7', label='S(b)')
f.edge('LR_2', 'LR_4', label='S(A)')
f.edge('LR_2', 'LR_5', label='SS(a)')
f.edge('LR_2', 'LR_6', label='SS(b)')
f.edge('LR_0', 'LR_1', label='SS(S)')
f.edge('LR_1', 'LR_3', label='S($end)')
f.edge('LR_0', 'LR_2', label='SS(B)')
f.edge('LR_7', 'LR_8', label='S(b)')
f.edge('LR_7', 'LR_5', label='S(a)')
f.edge('LR_8', 'LR_6', label='S(b)')
f.edge('LR_8', 'LR_5', label='S(a)')

f.view()
