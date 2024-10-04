from itertools import permutations

def calculateTotalCost(faciliteis: list, locations: list, assignment):
    totalCost = 0
    n = len(faciliteis)
    for i in range(n):
        for j in range(n):
            # compute cost for 2 facilities in 2 neighboring location
            f1 = assignment[i]
            f2 = assignment[j]
            l1 = i
            l2 = j

            totalCost += faciliteis[f1][f2] * locations[l1][l2]

    return totalCost

if __name__ == "__main__":
    # Facilities cost matrix
    facilities = [
        [0, 2, 3, 1],
        [2, 0, 1, 4],
        [3, 1, 0, 2],
        [1, 4, 2, 0]
    ]
 
    # Flow matrix
    locations = [
        [0, 1, 2, 3],
        [1, 0, 4, 2],
        [2, 4, 0, 1],
        [3, 2, 1, 0]
    ]
 
    n = len(facilities)

    minCost = float('inf')
    minAssignment = None
    for assignment in permutations(range(n)):
        totalCost = calculateTotalCost(facilities, locations, assignment)
        if totalCost < minCost:
            minCost = totalCost
            minAssignment = assignment

    # Print the optimal assignment and total cost
    print("Optimal Assignment: ", end="")
    for i in range(n):
        print(f"F{minAssignment[i] + 1}->L{i + 1} ", end="")
    print()
    print(f"Total Cost: {minCost}")