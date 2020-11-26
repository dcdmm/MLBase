from graphviz import Graph # 无向图

# 设置节点形状
f = Graph(name='example0', filename='example0.dot')

f.attr('node', shape='circle') # 圆
f.node('圆', fontname='SimHei')

f.attr('node', shape='doublecircle') # 双圆
f.node('双圆', fontname='SimHei')

f.attr('node', shape='egg') # 蛋状
f.node('蛋状', fontname='SimHei')

f.attr('node', shape='point') # 点
f.node('点', fontname='SimHei')

f.attr('node', shape='oval') # 椭圆
f.node('椭圆', fontname='SimHei')

f.attr('node', shape='box') # 方框
f.node('方框', fontname='SimHei')

f.attr('node', shape='polygon') # 多边形
f.node('多边形', fontname='SimHei')

f.attr('node', shape='diamond') # 菱形
f.node('菱形', fontname='SimHei')

f.attr('node', shape='trapezium') # 梯形
f.node('梯形', fontname='SimHei')

f.attr('node', shape='parallelogram') # 平行四边形
f.node('平行四边形', fontname='SimHei')

f.attr('node', shape='pentagon') # 五边形
f.node('五边形', fontname='SimHei')

f.attr('node', shape='hexagon') # 六边形
f.node('六边形', fontname='SimHei')

f.attr('node', shape='septagon') # 七边形
f.node('七边形', fontname='SimHei')

f.attr('node', shape='octagon') # 八边形
f.node('八边形', fontname='SimHei')

f.attr('node', shape='rectangle') # 长方形
f.node('长方形', fontname='SimHei')

f.attr('node', shape='square') # 正方形
f.node('正方形', fontname='SimHei')

f.attr('node', shape='triangle') # 三角形
f.node('三角形', fontname='SimHei')

f.attr('node', shape='triangle') # 星形
f.node('星形', fontname='SimHei')

f.attr('node', shape='rarrow') # 右箭头形
f.node('右箭头形', fontname='SimHei')

f.attr('node', shape='larrow') # 左箭头形
f.node('左箭头形', fontname='SimHei')

f.view()

