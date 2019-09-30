#encoding=utf8
import jieba

def read_in_graph(data_dir):

    nodes = []
    edges = {}

    cur_main = None

    with open(data_dir+'/kb/knowledge.txt','r') as fin:
        for idx, line in enumerate(fin):
            if line == '\n':
                cur_main = None
                continue
            if cur_main == None:
                cur_main = line.strip().split()
                if cur_main[0] not in nodes:
                    nodes.append(cur_main[0])
                if cur_main[1] not in edges:
                    edges[cur_main[1]] = []
            else:
                x = line.strip()
                x = ''.join(x.split())
                if x not in nodes:
                    nodes.append(x)

                if (cur_main[0], x) not in edges[cur_main[1]]:
                    edges[cur_main[1]].append((cur_main[0], x))
                if cur_main[1] in ['別名', '稱謂', '居住地', '兄弟姐妹', '主僕', '情人', '敵人']:
                    if (x, cur_main[0]) not in edges[cur_main[1]]:
                        edges[cur_main[1]].append((x, cur_main[0]))
                elif cur_main[1] == '兒女':
                    if '父母' not in edges:
                        edges['父母'] = []
                    if (x, cur_main[0]) not in edges['父母']:
                        edges['父母'].append((x, cur_main[0]))
                elif cur_main[1] == '父母':
                    if '兒女' not in edges:
                        edges['兒女'] = []
                    if (x, cur_main[0]) not in edges['兒女']:
                        edges['兒女'].append((x, cur_main[0]))

    nodes = list(sorted(nodes))
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
