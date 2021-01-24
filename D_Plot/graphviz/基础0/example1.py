import graphviz

with open('picture.dot', encoding='UTF-8') as f:
    dot_graph = f.read()

# 保存并查看
graphviz.Source(dot_graph).view() # Save the source to file, open the rendered result in a viewer.