visit = set()
graph = {
    'A' : ['B', 'C', 'D'],
    'B' : ['E', 'F'],
    'E' : ['G'],
    'F' : ['H'],
}
def dfs(graph, node, visit):
    if node not in visit:
        print(node)
        for neigh in graph[node]:
            dfs(graph, neigh, visit)

dfs(graph, '5', visit)



v = []
q = []

def bfs(graph, node, v):
    v.append(node)
    q.append(node)
     while q:
         m = q.pop(0)
         print(m, end = "")
         for neigh in graph[m]:
             if neigh not in v:
                v.append(node)
                q.append(node)

#bfs(graph, '5', v)