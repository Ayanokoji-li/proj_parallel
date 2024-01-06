import networkx as nx

# 从 "web-Google.mtx" 文件中读取图形
graph = nx.read_adjlist("web-Google.mtx", create_using=nx.DiGraph())

# 计算 PageRank
pagerank = nx.pagerank(graph)

# 打印 PageRank 值
for node, rank in pagerank.items():
    print(f"Node {node}: PageRank = {rank}")
