import copy

EPSILON = 1e-6

def calculate_consumption(problem, path):
    n = len(path)
    consumption = [0] * n
    consumption[0] = 0
    for i in range(1, n - 1):
        consumption[i] = consumption[i - 1] + problem.get_capacity(path[i])
    consumption[n - 1] = consumption[n - 2]
    return consumption

def calculate_distance_between_indices(problem, from_index, to_index):
    return problem.get_distance(from_index, to_index)

def calculate_path_distance(problem, path):
    sum_ = 0.0
    for i in range(1, len(path)):
        sum_ += calculate_distance_between_indices(problem, path[i - 1], path[i])
    return sum_

def calculate_solution_distance(problem, solution):
    total_distance = 0.0
    for path in solution:
        total_distance += calculate_path_distance(problem, path)
    return total_distance
