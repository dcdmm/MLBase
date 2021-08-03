import graphviz

# 可以是.dot或.gv文件
with open('picture_save.gv', encoding='UTF-8') as f:
    dot_graph = f.read()

# Verbatim DOT source code string to be rendered by Graphviz.
graphviz.Source(dot_graph, filename="picture1.gv").view()
