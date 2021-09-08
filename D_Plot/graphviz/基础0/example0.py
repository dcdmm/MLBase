from graphviz import Digraph  # 有向图,无向图为Graph

dot = Digraph(name='picture',  # graph name used in the source code.
              filename='picture.dot',  # Filename for saving the source (defaults to name + '.gv')
              format='pdf')  # Rendering output format ('pdf', 'png', …).
dot.node(name='A', label='作者', fontname="KaiTi")  # 楷体(中文必须设置fontname为楷体,微软雅黑,宋体,黑体)
dot.node('B', '医生', fontname='Microsoft YaHei')  # 微软雅黑
dot.node('C', 'teach')
dot.edge(tail_name='B', head_name='A', label='箭头1', fontname='SimHei')  # Create an edge between two nodes
dot.edge('C', 'A', label='箭头2', fontname='SimHei', fontcolor='red', fontsize='5')  # 黑体(红色5,5榜)
dot.edge('A', 'B', label='箭头3', fontname='SimSun')  # 宋体

dot.attr(label=r'\n\nEntity Relation Diagram\ndrawn by NEATO', fontcolor='green', fontsize='20')  # 图片标题
print(type(dot))

# Save the source to file, open the rendered result in a viewer.
# filename – Filename for saving the source (defaults to name + '.gv')
dot.view(filename="picture_view.gv")  # 生成picture_view.gv和picture_view.gv.pdf

# Save the DOT source to file. Ensure the file ends with a newline.
# filename – Filename for saving the source (defaults to name + '.gv')
dot.save(filename='picture_save.gv')  # 生成picture_view_gv
