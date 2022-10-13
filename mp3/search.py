# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze
def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


def astar(maze, ispart1=False):
    """
    This function returns an optimal path in a list, which contains the start and objective.

    @param maze: Maze instance from maze.py
    @param ispart1:pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    @return: a path in the form of a list of MazeState objects. If there is no path, return None.
    """
    # Your code here
    starting_state = maze.getStart()
    visited_states = {starting_state: (None, 0)}
    fully_explored_states = set()
    frontier = []
    heapq.heappush(frontier, starting_state)
    goal_state = None
    while goal_state is None and len(frontier):
        current_state = heapq.heappop(frontier)
        neighbors = current_state.get_neighbors(ispart1)
        if all([neighbor in visited_states for neighbor in neighbors]):
            fully_explored_states.add(current_state)
            continue
        for neighbor in neighbors:
            if neighbor in fully_explored_states:
                continue
            total_distance = neighbor.dist_from_start + neighbor.h
            if neighbor in visited_states and visited_states[neighbor][1] <= total_distance:
                continue

            visited_states[neighbor] = current_state, total_distance
            heapq.heappush(frontier, neighbor)
            if neighbor.is_goal():
                goal_state = neighbor
                break
    return backtrack(visited_states, goal_state) if goal_state is not None else None


# This is the same as backtrack from MP2
def backtrack(visited_states, current_state):
    path = []
    # Your code here ---------------
    while True:
        path.append(current_state)
        parent = visited_states[current_state][0]
        if parent is not None:
            current_state = parent
        else:
            break
    # ------------------------------
    return path[::-1]
        