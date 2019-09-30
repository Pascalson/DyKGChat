from shortest_paths import *

class SubGraph():
    def __init__(self, nodes, edges):
        self.max_reasoning_hops = 10
        self.num_shortest_paths = 5
        self.global_nodes = nodes
        self.global_Tedges = self._trans_edges(edges)
        self.sp_op = SSSPs(nodes, edges)
        self.triples = []

    def _trans_edges(self, edges):
        #TODO: only for each pair has only **one** relation
        Tedges = {}
        for key, value in edges.items():
            for v in value:
                if v[0] not in Tedges:
                    Tedges[v[0]] = {}
                if v[1] not in Tedges[v[0]]:
                    Tedges[v[0]][v[1]] = []
                Tedges[v[0]][v[1]].append(key)
        return Tedges


    def reset(self):
        self.triples = []
    
    def sample(self, sources, targets):
        self.reset()
        for t in targets:
            for s in sources:
                paths, dists = self.sp_op.run(s, t, self.num_shortest_paths)
                for pair in paths:
                    if pair[0] == pair[1]:
                        continue
                    triple = (pair[0], self.global_Tedges[pair[0]][pair[1]][-1], pair[1])
                    if triple not in self.triples:
                        self.triples.append(triple)
