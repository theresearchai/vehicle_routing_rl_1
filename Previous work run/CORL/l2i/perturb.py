from utils import *
import numpy as np
import copy

def construct_solution(problem, config, existing_solution=None, step=0):
    solution = []
    n = problem.get_num_customers()
    customer_indices = np.arange(n + 1)

    if (existing_solution is not None) and (config.num_paths_to_ruin != -1):
        distance = calculate_solution_distance(problem, existing_solution)
        min_reconstructed_distance = float('inf')
        solution_to_return = None
        for _ in range(1):
            reconstructed_solution = reconstruct_solution(problem, existing_solution, step, config)
            reconstructed_distance = calculate_solution_distance(problem, reconstructed_solution)
            if reconstructed_distance / distance <= 1.05:
                solution_to_return = reconstructed_solution
                break
            else:
                if reconstructed_distance < min_reconstructed_distance:
                    min_reconstructed_distance = reconstructed_distance
                    solution_to_return = reconstructed_solution
        return solution_to_return
    else:
        start_customer_index = 1

    trip = [0]
    capacity_left = problem.get_capacity(0)
    i = start_customer_index
    while i <= n:
        random_index = np.random.randint(low=i, high=n+1)

        to_indices = []
        adjusted_distances = []
        for j in range(i, n + 1):
            if problem.get_capacity(customer_indices[j]) > capacity_left:
                continue
            to_indices.append(j)
            adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
        random_index = sample_next_index(to_indices, adjusted_distances)

        if random_index == 0 or capacity_left < problem.get_capacity(customer_indices[random_index]):
            trip.append(0)
            solution.append(trip)
            trip = [0]
            capacity_left = problem.get_capacity(0)
            continue
        customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
        trip.append(customer_indices[i])
        capacity_left -= problem.get_capacity(customer_indices[i])
        i += 1
    if len(trip) > 1:
        trip.append(0)
        solution.append(trip)
    solution.append([0, 0])

    problem.reset_change_at_and_no_improvement_at()
    return solution

def reconstruct_solution(problem, existing_solution, step, config):
    distance_hash = round(calculate_solution_distance(problem, existing_solution) * 1e6)
    if config.detect_negative_cycle and distance_hash not in problem.distance_hashes:
        problem.add_distance_hash(distance_hash)
        positive_cycles = []
        cycle_selected = None
        for capacity in range(1, 10):
            graph = construct_graph(problem, existing_solution, capacity)
            negative_cycle, flag = graph.find_negative_cycle()
            if negative_cycle:
                if flag == -1.0:
                    cycle_selected = negative_cycle
                    break
                else:
                    positive_cycles.append(negative_cycle)
        if cycle_selected is None and len(positive_cycles) > 0:
            index = np.random.choice(range(len(positive_cycles)), 1)[0]
            cycle_selected = positive_cycles[index]
        if cycle_selected is not None:
                negative_cycle = cycle_selected
                improved_solution = copy.deepcopy(existing_solution)
                customers = []
                for pair in negative_cycle:
                    path_index, node_index = pair[0], pair[1]
                    customers.append(improved_solution[path_index][node_index])
                customers = [customers[-1]] + customers[:-1]
                for index in range(len(negative_cycle)):
                    pair = negative_cycle[index]
                    path_index, node_index = pair[0], pair[1]
                    improved_solution[path_index][node_index] = customers[index]
                    problem.mark_change_at(step, [path_index])
                return improved_solution

    solution = []
    n = problem.get_num_customers()
    customer_indices = np.arange(n + 1)

    candidate_indices = []
    for path_index in range(len(existing_solution)):
        if len(existing_solution[path_index]) > 2:
            candidate_indices.append(path_index)
    paths_ruined = np.random.choice(candidate_indices, config.num_paths_to_ruin, replace=False)
    start_customer_index = n + 1
    for path_index in paths_ruined:
        path = existing_solution[path_index]
        for customer_index in path:
            if customer_index == 0:
                continue
            start_customer_index -= 1
            customer_indices[start_customer_index] = customer_index

    if np.random.uniform() < 0.5:
        while len(solution) == 0:
            paths_ruined = np.random.choice(candidate_indices, config.num_paths_to_ruin, replace=False)
            solution = reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined)
    else:
        trip = [0]
        capacity_left = problem.get_capacity(0)
        i = start_customer_index
        while i <= n:
            to_indices = []
            adjusted_distances = []
            for j in range(i, n + 1):
                if problem.get_capacity(customer_indices[j]) > capacity_left:
                    continue
                to_indices.append(j)
                adjusted_distances.append(
                    calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
            random_index = sample_next_index(to_indices, adjusted_distances)

            if random_index == 0:
                trip.append(0)
                solution.append(trip)
                trip = [0]
                capacity_left = problem.get_capacity(0)
                continue
            customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
            trip.append(customer_indices[i])
            capacity_left -= problem.get_capacity(customer_indices[i])
            i += 1
        if len(trip) > 1:
            trip.append(0)
            solution.append(trip)

    while len(solution) < len(paths_ruined):
        solution.append([0, 0])
    improved_solution = copy.deepcopy(existing_solution)
    solution_index = 0
    for path_index in sorted(paths_ruined):
        improved_solution[path_index] = copy.deepcopy(solution[solution_index])
        solution_index += 1
    problem.mark_change_at(step, paths_ruined)
    for solution_index in range(len(paths_ruined), len(solution)):
        improved_solution.append(copy.deepcopy(solution[solution_index]))
        problem.mark_change_at(step, [len(improved_solution) - 1])

    has_seen_empty_path = False
    for path_index in range(len(improved_solution)):
        if len(improved_solution[path_index]) == 2:
            if has_seen_empty_path:
                empty_slot_index = path_index
                for next_path_index in range(path_index + 1, len(improved_solution)):
                    if len(improved_solution[next_path_index]) > 2:
                        improved_solution[empty_slot_index] = copy.deepcopy(improved_solution[next_path_index])
                        empty_slot_index += 1
                improved_solution = improved_solution[:empty_slot_index]
                problem.mark_change_at(step, list(range(path_index, empty_slot_index)))
                break
            else:
                has_seen_empty_path = True
    return improved_solution

def reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined):
    path0 = copy.deepcopy(existing_solution[paths_ruined[0]])
    path1 = copy.deepcopy(existing_solution[paths_ruined[1]])
    num_exchanged = 0
    for i in range(1, len(path0) - 1):
        for j in range(1, len(path1) - 1):
            if problem.get_capacity(path0[i]) == problem.get_capacity(path1[j]):
                #TODO
                if problem.get_distance(path0[i], path1[j]) < 0.2:
                    path0[i], path1[j] = path1[j], path0[i]
                    num_exchanged += 1
                    break
    if num_exchanged >= 0:
        return [path0, path1]
    else:
        return []

def calculate_adjusted_distance_between_indices(problem, from_index, to_index):
    distance = problem.get_distance(from_index, to_index)
    frequency = problem.get_frequency(from_index, to_index)
    return distance * (1.0 - frequency)

def calculate_replacement_cost(problem, from_index, to_indices):
    return problem.get_distance(from_index, to_indices[0]) + problem.get_distance(from_index, to_indices[2]) \
        - problem.get_distance(to_indices[1], to_indices[0]) - problem.get_distance(to_indices[1], to_indices[2])

def sample_next_index(to_indices, adjusted_distances):
    if len(to_indices) == 0:
        return 0
    adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
    adjusted_probabilities /= np.sum(adjusted_probabilities)
    return np.random.choice(to_indices, p=adjusted_probabilities)

class Graph:
    def __init__(self, problem, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for from_index in range(self.num_nodes):
            for to_index in range(from_index + 1, self.num_nodes):
                self.distance_matrix[from_index][to_index] = calculate_replacement_cost(problem, nodes[from_index][1], nodes[to_index])
                self.distance_matrix[to_index][from_index] = calculate_replacement_cost(problem, nodes[to_index][1], nodes[from_index])

    def find_negative_cycle(self):
        distance = [float('inf')] * self.num_nodes
        predecessor = [None] * self.num_nodes
        source = 0
        distance[source] = 0.0

        for i in range(1, self.num_nodes):
            improved = False
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    w = self.distance_matrix[u][v]
                    if distance[u] + w < distance[v]:
                        distance[v] = distance[u] + w
                        predecessor[v] = u
                        improved = True
            if not improved:
                break

        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                w = self.distance_matrix[u][v]
                if distance[u] + w + EPSILON < distance[v]:
                    visited = [0] * self.num_nodes
                    negative_cycle = []
                    negative_cycle.append(self.nodes[v][-2:])
                    count = 1
                    while (u != v) and (not visited[u]):
                        negative_cycle.append(self.nodes[u][-2:])
                        visited[u] = count
                        count += 1
                        u = predecessor[u]
                    if u != v:
                        negative_cycle = negative_cycle[visited[u]:]
                    return negative_cycle[::-1], -1.0

        num_cyclic_perturb = 4
        cutoff = 0.3
        if self.num_nodes >= num_cyclic_perturb:
            candidate_cycles = []
            for index in range(self.num_nodes):
                candidate_cycles.append(([index], 0.0))
            for index_to_choose in range(1, num_cyclic_perturb):
                next_candidate_cycles = []
                for cycle in candidate_cycles:
                    nodes = cycle[0]
                    total_distance = cycle[1]
                    for index in range(self.num_nodes):
                        if index not in nodes:
                            if index_to_choose == num_cyclic_perturb - 1:
                                new_total_distance = total_distance + self.distance_matrix[nodes[-1]][index] + self.distance_matrix[index][nodes[0]]
                            else:
                                new_total_distance = total_distance + self.distance_matrix[nodes[-1]][index]
                            if new_total_distance < cutoff:
                                next_candidate_cycles.append((nodes + [index], new_total_distance))
                candidate_cycles = next_candidate_cycles
            if len(candidate_cycles) > 0:
                random_indices = np.random.choice(range(len(candidate_cycles)), 1)[0]
                random_indices = candidate_cycles[random_indices][0]
                negative_cycle = []
                for u in random_indices:
                    negative_cycle.append(self.nodes[u][-2:])
                return negative_cycle, 1.0
        return None, None

def construct_graph(problem, solution, capacity):
    nodes = []
    for path_index in range(len(solution)):
        path = solution[path_index]
        if len(path) > 2:
            node_index = 1
            while node_index < len(path) - 1:
                node_index_end = node_index + 1
                if problem.get_capacity(path[node_index]) == capacity:
                    while problem.get_capacity(path[node_index_end]) == capacity:
                        node_index_end += 1
                    sampled_node_index = np.random.choice(range(node_index, node_index_end))
                    nodes.append([path[sampled_node_index - 1], path[sampled_node_index], path[sampled_node_index + 1],
                                  path_index, sampled_node_index])
                node_index = node_index_end
    graph = Graph(problem, nodes)
    return graph
