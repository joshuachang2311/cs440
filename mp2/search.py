import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search 
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state)
    #   - keep track of the distance of each state from start
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    goal_state = None
    while len(frontier):
        current_state = heapq.heappop(frontier)
        if current_state.is_goal():
            goal_state = current_state
            break
        # if current_state in fully_explored_states:
        #     continue
        # if all([neighbor in visited_states for neighbor in neighbors]):
        #     fully_explored_states.add(current_state)
        for neighbor in current_state.get_neighbors():
            total_distance = neighbor.dist_from_start + neighbor.h
            if neighbor in visited_states and visited_states[neighbor][1] < total_distance:
                pass
            else:
                visited_states[neighbor] = current_state, total_distance
                heapq.heappush(frontier, neighbor)
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return backtrack(visited_states, goal_state) if goal_state is not None else []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------
    current_state = goal_state
    while True:
        path.append(current_state)
        parent = visited_states[current_state][0]
        if parent is not None:
            current_state = parent
        else:
            break
    # ------------------------------
    return path[::-1]
