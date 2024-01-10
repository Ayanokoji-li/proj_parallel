import networkx as nx

# 从 "web-Google.mtx" 文件中读取图形
graph = nx.read_adjlist("data/web-Google.mtx", create_using=nx.DiGraph(), comments="%")

# 计算 PageRank
pagerank = nx.pagerank(graph)

# 打印 PageRank 值
i = 0
for node, rank in pagerank.items():
    print(f"Node {node}: PageRank = {rank}")
    i+=1
    if(i == 10):
        break
