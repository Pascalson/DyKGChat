import jieba

def read_in_graph(data_dir):
    nodes = []
    edges = {}
    #TODO: implement this function for different dataset
    return nodes, edges

if __name__ == '__main__':
    nodes, edges = read_in_graph('.')
    print(nodes)
    print(len(nodes))
    print(edges.keys())
    total_edges = 0
    for key, value in edges.items():
        total_edges += len(value)
    for key, value in edges.items():
        print('{}:{}'.format(key, len(value)/total_edges))
