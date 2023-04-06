import random

TSP = [
        [0, 400, 500, 300],
        [400, 0, 300, 500],
        [500, 300, 0, 400],
        [300, 500, 400, 0]
    ]


def randomSol(tsp):
    cities = list(range(len(tsp)))
    solution = []

    for i in range(len(tsp)):
        randCity = cities[random.randint(0, len(cities) - 1)]
        solution.append(randCity)
        cities.remove(randCity)

    return solution


def routeLen(tsp, solution):
    routeLen = 0
    for i in range(len(solution)):
        routeLen += tsp[solution[i - 1]][solution[i]]
    return routeLen


def getNeighbours(solution):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            temp = solution.copy()
            temp[i] = solution[j]
            temp[j] = solution[i]
            neighbours.append(temp)
    return neighbours


def hillClimbing(tsp):
    currentSolution = randomSol(tsp)
    currentRouteLength = routeLen(tsp, currentSolution)
    neighbours = getNeighbours(currentSolution)
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    while bestNeighbourRouteLength < currentRouteLength:
        currentSolution = bestNeighbour
        currentRouteLength = bestNeighbourRouteLength
        neighbours = getNeighbours(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    return currentSolution, currentRouteLength


def getBestNeighbour(tsp, neighbours):
    bestRouteLength = routeLen(tsp, neighbours[0])
    bestNeighbour = neighbours[0]
    for neighbour in neighbours:
        currentRouteLength = routeLen(tsp, neighbour)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength

print(hillClimbing(TSP))
