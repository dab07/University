import sys
import numpy as np

def solution(n, m, queries):
    mat = []
    ans = []
    for i in range(1, n + 1):
        temp = []
        for j in range(1, m + 1):
            temp.append(i * j)
        mat.append(temp)
    mat = np.array(mat)

    print(mat)
    mat = np.delete(mat, 0, axis=1)
    print(mat)

    for i in range(len(queries)):
        if queries[i] == [0]:
            ans.append(mini(mat))
        elif queries[i][0] == 1:
            # for j in range(len(mat[0])):
            #     mat[queries[i][1] - 1][j] = sys.maxsize
            mat = np.delete(mat, 0, axis=queries[i][1] - 1)
        elif queries[i][0] == 2:
            # for j in range(len(mat)):
            #     mat[j][queries[i][1] - 1] = sys.maxsize
            mat = np.delete(mat, 1, axis=queries[i][1] - 1)

    return ans

def mini(m):
    minm = m[0][0]
    for i in range(len(m)):
        for j in range(len(m[0])):
            minm = min(minm, m[i][j])
    return minm


n = 5
m = 1
queries = [[1,3],
    [1,2],
    [1,4],
    [0],
    [1,1],
    [0]]
ans = solution(n, m, queries)
print(ans)