import numpy as np

def construct_maze(maze_id=0, length=1):
    # define the maze to use
    if maze_id == 0:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 1:  # donuts maze: can reach the single goal by 2 equal paths
        c = length + 4
        M = np.ones((c, c))
        M[1:c - 1, (1, c - 2)] = 0
        M[(1, c - 2), 1:c - 1] = 0
        M = M.astype(int).tolist()
        M[1][c // 2] = 'r'
        M[c - 2][c // 2] = 'g'
        structure = M

    elif maze_id == 2:  # spiral maze: need to use all the keys (only makes sense for length >=3)
        c = length + 4
        M = np.ones((c, c))
        M[1:c - 1, (1, c - 2)] = 0
        M[(1, c - 2), 1:c - 1] = 0
        M = M.astype(int).tolist()
        M[1][c // 2] = 'r'
        # now block one of the ways and put the goal on the other side
        M[1][c // 2 - 1] = 1
        M[1][c // 2 - 2] = 'g'
        structure = M

    elif maze_id == 3:  # corridor with goals at the 2 extremes
        structure = [
            [1] * (2 * length + 5),
            [1, 'g'] + [0] * length + ['r'] + [0] * length + ['g', 1],
            [1] * (2 * length + 5),
            ]

    elif 4 <= maze_id <= 7:  # cross corridor, goal in
        c = 2 * length + 5
        M = np.ones((c, c))
        M = M - np.diag(np.ones(c))
        M = M - np.diag(np.ones(c - 1), 1) - np.diag(np.ones(c - 1), -1)
        i = np.arange(c)
        j = i[::-1]
        M[i, j] = 0
        M[i[:-1], j[1:]] = 0
        M[i[1:], j[:-1]] = 0
        M[np.array([0, c - 1]), :] = 1
        M[:, np.array([0, c - 1])] = 1
        M = M.astype(int).tolist()
        M[c // 2][c // 2] = 'r'
        if maze_id == 4:
            M[1][1] = 'g'
        if maze_id == 5:
            M[1][c - 2] = 'g'
        if maze_id == 6:
            M[c - 2][1] = 'g'
        if maze_id == 7:
            M[c - 2][c - 2] = 'g'
        structure = M

    elif maze_id == 8:  # reflexion of benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

    elif maze_id == 9:  # sym benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 'r', 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1],
        ]

    elif maze_id == 10:  # reflexion of sym of benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 'g', 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 'r', 1],
            [1, 1, 1, 1, 1],
        ]
    if structure:
        return structure
    else:
        raise NotImplementedError("The provided MazeId is not recognized")
