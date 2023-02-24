import heapq
import numpy as np

def print_succ(state):
    successors = succ(state)

    for i in successors:
        print(i, end=" ")
        h = get_heuristic(i)
        print("h=" + str(h))

def succ(state):

    index = 0
    for i in range(len(state)): #get index of 0
        if state[i] == 0:
            index = i

    successors = []

    if index % 3 == 0: #(0, 3, 6)
        successors.append(get_right_succ(state, index)) #adds right
        if index + 3 != 3: #adds above if (3, 6)
            successors.append(get_above_succ(state, index))
        if index - 3 != 3: #adds below if (0, 3)
            successors.append(get_below_succ(state, index))

    elif index % 3 == 1: #(1, 4, 7)
        successors.append(get_right_succ(state, index)) #adds right
        successors.append(get_left_succ(state, index)) #adds left
        if index + 3 != 4: #adds above if (4, 7)
            successors.append(get_above_succ(state, index))
        if index - 3 != 4: #adds below if (1, 4)
            successors.append(get_below_succ(state, index))

    else: #(2, 5, 8)
        successors.append(get_left_succ(state, index)) #adds left
        if index + 3 != 5: #adds above if (5, 8)
            successors.append(get_above_succ(state, index))
        if index - 3 != 5: #adds below if (2, 5)
            successors.append(get_below_succ(state, index))

    return sorted(successors)

def get_left_succ(state, index):
    succ = state.copy() #adds left
    succ[index], succ[index-1] = state[index-1], state[index]
    return succ

def get_right_succ(state, index):
    succ = state.copy() #adds right
    succ[index], succ[index+1] = state[index+1], state[index]
    return succ

def get_above_succ(state, index):
    succ = state.copy()
    succ[index], succ[index-3] = state[index-3], state[index]
    return succ

def get_below_succ(state, index):
    succ = state.copy()
    succ[index], succ[index+3] = state[index+3], state[index]
    return succ

def get_heuristic(state):
    h = 0
    succ = np.array(state).reshape(3, 3) #convert into 3x3 for ease of operation
    for i in range(len(succ)):
        for j in range(len(succ[i])):
            a = succ[i][j]
            if a == 1:
                dist = abs(i - 0) + abs(j - 0)
                h += dist
            elif a == 2:
                dist = abs(i - 0) + abs(j - 1)
                h += dist
            elif a == 3:
                dist = abs(i - 0) + abs(j - 2)
                h += dist
            elif a == 4:
                dist = abs(i - 1) + abs(j - 0)
                h += dist
            elif a == 5:
                dist = abs(i - 1) + abs(j - 1)
                h += dist
            elif a == 6:
                dist = abs(i - 1) + abs(j - 2)
                h += dist
            elif a == 7:
                dist = abs(i - 2) + abs(j - 0)
                h += dist
            elif a == 8:
                dist = abs(i - 2) + abs(j - 1)
                h += dist
    return h

def terminal(state): #checks if a given state is terminal
    win = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    count = 0
    for i in range(len(state)):
        if state[i] == win[i]:
            count += 1
    if count == 9:
        return True
    return False

def solve(state):
    Open = []
    Closed = []
    explored = []

    h = get_heuristic(state)
    #(g+h, state, (g, h, parent_index))
    heapq.heappush(Open, (h, state, (0, h, -1)))

    while len(Open) > 0:
        b = heapq.heappop(Open)
        g = b[2][0]
        p = b[2][2]
        h = get_heuristic(b[1])

        if b[1] in explored: #do not explore already explored states
            continue
        explored.append(b[1])

        if terminal(b[1]): #if the curState is terminal, get the path
            reverse = []
            prev = b[1]

            #iterate backwards through Closed, finding a parent state related to the previous
            for j in range(p): #needs to loop multiple times to find the full path
                for i in range(len(Closed)-1, -1, -1):
                    if Closed[i][2][2] == p-1 and Closed[i][1] in succ(prev):
                        reverse.append(Closed[i])
                        p = Closed[i][2][2]
                        prev = Closed[i][1]

            #print start state, needed to be separate from path
            #otherwise the start state might not get printed at all for some reason
            print(state, end=" ")
            print("h=" + str(get_heuristic(state)) + " moves: " + str(0))

            path = reversed(reverse)
            for i in path:
                if i[1] != state: #make sure we aren't reprinting the start
                    print(i[1], end=" ")
                    print("h=" + str(i[2][1]) + " moves: " + str(i[2][0]))

            if b[1] != state: #print terminal state unless it's the starting state
                print(b[1], end=" ")
                print("h=" + str(b[2][1]) + " moves: " + str(b[2][0]))

            break

        heapq.heappush(Closed, (h+g, b[1], (g, h, p)))
        successors = succ(b[1])
        for child in successors:
            if child not in explored: #do not look at explored children
                gChild = g + 1
                pChild = p + 1
                hChild = get_heuristic(child)
                heapq.heappush(Open, (hChild+gChild, child, (gChild, hChild, pChild)))
    #print("Max queue length: " + str(len(Open)))

"""
if __name__=='__main__':
    print("Initial state:")
    print([8, 6, 7, 2, 5, 4, 3, 0, 1])
    print("\nSuccessors:")
    print_succ([8, 6, 7, 2, 5, 4, 3, 0, 1])
    print("\nPath:")
    solve([8, 6, 7, 2, 5, 4, 3, 0, 1])
"""
