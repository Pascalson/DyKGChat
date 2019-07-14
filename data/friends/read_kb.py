
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
                cur_main = line.strip().split(' | ')
                cur_main[0] = '_'.join(cur_main[0].split())#FIXME
                if cur_main[0] not in nodes:
                    nodes.append(cur_main[0])
                if cur_main[1] not in edges:
                    edges[cur_main[1]] = []
            else:
                if ' | ' in line:
                    x, mark = line.strip().split(' | ')
                else:
                    x = line
                x = x.strip()
                if ',' in x:
                    x1, x2 = x.split(', ')
                    x1 = '_'.join(x1.split())
                    x2 = '_'.join(x2.split())
                    x = '_'.join([x1,x2])
                    if x1 not in nodes:
                        nodes.append(x1)
                    if x2 not in nodes:
                        nodes.append(x2)
                    if (x, x2) not in edges['name']:
                        edges['name'].append((x1, x2))
                    if (x2, x) not in edges['name']:
                        edges['name'].append((x2, x1))
                    if (x1, x) not in edges['name']:
                        edges['name'].append((x1, x))
                    if (x, x1) not in edges['name']:
                        edges['name'].append((x, x1))
                else:
                    x = '_'.join(x.split())

                if x not in nodes:
                    nodes.append(x)

                cur_main[0] = '_'.join(cur_main[0].split())
                if (cur_main[0], x) not in edges[cur_main[1]]:
                    edges[cur_main[1]].append((cur_main[0], x))
                if cur_main[1] in ['name', 'nickname', 'alias', 'occupation', 'spouse', 'lover', 'friend', 'cousin']:
                    if (x, cur_main[0]) not in edges[cur_main[1]]:
                        edges[cur_main[1]].append((x, cur_main[0]))
                elif cur_main[1] in ['father','mother','daughter','son','grandmother','grandaunt','sister','brother','uncle','aunt','niece','nephew','pet']:
                    if mark not in edges:
                        edges[mark] = []
                    if (x, cur_main[0]) not in edges[mark]:
                        edges[mark].append((x, cur_main[0]))

    nodes = list(sorted(nodes))
    return nodes, edges

if __name__ == '__main__':
    nodes, edges = read_in_graph('.')
    print(len(nodes))
    print(edges.keys())
