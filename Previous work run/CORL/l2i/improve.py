from utils import *
import copy

def do_two_opt_path(path, first, second):
    improved_path = copy.deepcopy(path)
    first = first + 1
    while first < second:
        improved_path[first], improved_path[second] = improved_path[second], improved_path[first]
        first = first + 1
        second = second - 1
    return improved_path


def two_opt_path(problem, path):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(n - 1):
        for second in range(first + 2, n):
            before = calculate_distance_between_indices(problem, path[first], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
            after = calculate_distance_between_indices(problem, path[first], path[second]) \
                    + calculate_distance_between_indices(problem, path[first + 1], path[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return do_two_opt_path(path, label[0], label[1]), min_delta, label


def do_exchange_path(path, first, second):
    improved_path = copy.deepcopy(path)
    improved_path[first], improved_path[second] = improved_path[second], improved_path[first]
    return improved_path


def exchange_path(problem, path):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(1, n - 1):
        for second in range(first + 1, n):
            if second == first + 1:
                before = calculate_distance_between_indices(problem, path[first - 1], path[first]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
                after = calculate_distance_between_indices(problem, path[first - 1], path[second]) \
                     + calculate_distance_between_indices(problem, path[first], path[second + 1])
            else:
                before = calculate_distance_between_indices(problem, path[first - 1], path[first]) \
                     + calculate_distance_between_indices(problem, path[first], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second - 1], path[second]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
                after = calculate_distance_between_indices(problem, path[first - 1], path[second]) \
                     + calculate_distance_between_indices(problem, path[second], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second - 1], path[first]) \
                     + calculate_distance_between_indices(problem, path[first], path[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return do_exchange_path(path, label[0], label[1]), min_delta, label

def do_relocate_path(path, first, first_tail, second):
    segment = path[first:(first_tail + 1)]
    improved_path = path[:first] + path[(first_tail + 1):]
    if second > first_tail:
        second -= (first_tail - first + 1)
    return improved_path[:(second + 1)] + segment + improved_path[(second + 1):]


def relocate_path(problem, path, exact_length=1):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(1, n - exact_length + 1):
        first_tail = first + exact_length - 1
        for second in range(n):
            if second >= first - 1 and second <= first_tail:
                continue
            before = calculate_distance_between_indices(problem, path[first - 1], path[first]) \
                    + calculate_distance_between_indices(problem, path[first_tail], path[first_tail + 1]) \
                    + calculate_distance_between_indices(problem, path[second], path[second + 1])
            after = calculate_distance_between_indices(problem, path[first - 1], path[first_tail + 1]) \
                    + calculate_distance_between_indices(problem, path[second], path[first]) \
                    + calculate_distance_between_indices(problem, path[first_tail], path[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, first_tail, second
    if label is None:
        return None, None, None
    else:
        return do_relocate_path(path, label[0], label[1], label[2]), min_delta, label

def do_cross_two_paths(path_first, path_second, first, second):
    return path_first[:(first + 1)] + path_second[(second + 1):], path_second[:(second + 1)] + path_first[(first + 1):]


def cross_two_paths(problem, path_first, path_second):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)

    start_of_second_index = 0
    for first in range(n_first):
        capacity_from_first_to_second = consumed_capacities_first[n_first - 1] - consumed_capacities_first[first]
        for second in range(start_of_second_index, n_second):
            if consumed_capacities_second[second] + capacity_from_first_to_second > problem.get_capacity(path_second[0]):
                break
            if consumed_capacities_first[first] + (consumed_capacities_second[n_second - 1] - consumed_capacities_second[second]) > problem.get_capacity(path_first[0]):
                start_of_second_index = second + 1
                continue
            before = calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
            after = calculate_distance_between_indices(problem, path_first[first], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_first[first + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_cross_two_paths(path_first, path_second, label[0], label[1])
        return improved_path_first, improved_path_second, min_delta, label


def do_relocate_two_paths(path_first, path_second, first, first_tail, second):
    return path_first[:first] + path_first[(first_tail + 1):], \
           path_second[:(second + 1)] + path_first[first:(first_tail + 1)] + path_second[(second + 1):]


def relocate_two_paths(problem, path_first, path_second, exact_length=None):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)

    max_length = 1
    min_length = 1
    if exact_length:
        max_length = exact_length
        min_length = exact_length
    for first in range(1, n_first):
        for first_tail in range((first + min_length - 1), min(first + max_length, n_first)):
            capacity_difference = (consumed_capacities_first[first_tail] - consumed_capacities_first[first - 1])
            if consumed_capacities_second[n_second - 1] + capacity_difference > problem.get_capacity(path_second[0]):
                break
            for second in range(0, n_second):
                before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
                after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second], path_first[first])\
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second + 1])
                delta = after - before
                if delta < min_delta:
                    min_delta = delta
                    label = first, first_tail, second
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_relocate_two_paths(path_first, path_second, label[0], label[1], label[2])
        return improved_path_first, improved_path_second, min_delta, label


def do_exchange_two_paths(path_first, path_second, first, first_tail, second, second_tail):
    return path_first[:first] + path_second[second:(second_tail + 1)] + path_first[(first_tail + 1):], \
           path_second[:second] + path_first[first:(first_tail + 1)] + path_second[(second_tail + 1):]


def exchange_two_paths(problem, path_first, path_second, exact_lengths=None):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)
    if exact_lengths:
        min_length_first, max_length_first = exact_lengths[0], exact_lengths[0]
        min_length_second, max_length_second = exact_lengths[1], exact_lengths[1]
    else:
        min_length_first, max_length_first = 1, 1
        min_length_second, max_length_second = 1, 1

    min_delta = -EPSILON
    label = None
    all_delta = 0.0
    for first in range(1, n_first):
        for first_tail in range((first + min_length_first - 1), min(first + max_length_first, n_first)):
            if first_tail >= n_first:
                break
            for second in range(1, n_second):
                if first_tail >= n_first:
                    break
                for second_tail in range((second + min_length_second - 1), min(second + max_length_second, n_second)):
                    if first_tail >= n_first:
                        break
                    if second_tail >= n_second:
                        break
                    capacity_difference = (consumed_capacities_first[first_tail] - consumed_capacities_first[first - 1]) - \
                                          (consumed_capacities_second[second_tail] - consumed_capacities_second[second - 1])
                    if consumed_capacities_first[n_first - 1] - capacity_difference <= problem.get_capacity(path_first[0]) and \
                            consumed_capacities_second[n_second - 1] + capacity_difference <= problem.get_capacity(path_second[0]):
                        pass
                    else:
                        continue
                    before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second])\
                     + calculate_distance_between_indices(problem, path_second[second_tail], path_second[second_tail + 1])
                    after = calculate_distance_between_indices(problem, path_first[first - 1], path_second[second]) \
                     + calculate_distance_between_indices(problem, path_second[second_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first])\
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second_tail + 1])
                    delta = after - before
                    if delta < -EPSILON:
                        all_delta += delta
                        label = first, first_tail, second, second_tail
                        path_first, path_second = do_exchange_two_paths(path_first, path_second, label[0], label[1], label[2], label[3])
                        #TODO(xingwen): speedup
                        n_first = len(path_first) - 1
                        n_second = len(path_second) - 1
                        consumed_capacities_first = calculate_consumption(problem, path_first)
                        consumed_capacities_second = calculate_consumption(problem, path_second)
    if label is None:
        return None, None, None, None
    else:
        return path_first, path_second, all_delta, label


def do_eject_two_paths(path_first, path_second, first, second):
    return path_first[:first] + path_first[(first + 1):], \
           path_second[:second] + path_first[first:(first + 1)] + path_second[(second + 1):]


def eject_two_paths(problem, path_first, path_second):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = float("inf")
    label = None
    consumed_capacities_second = calculate_consumption(problem, path_second)

    for first in range(1, n_first):
        for second in range(1, n_second):
            capacity_difference = problem.get_capacity(path_first[first]) - problem.get_capacity(path_second[second])
            if consumed_capacities_second[n_second - 1] + capacity_difference > problem.get_capacity(path_second[0]):
                continue
            before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second]) \
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
            after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_second[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second, path_second[second]
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_eject_two_paths(path_first, path_second, label[0], label[1])
        return improved_path_first, improved_path_second, min_delta, label[2]


def insert_into_path(path, first, problem):
    n = len(path) - 1
    min_delta = float("inf")
    label = None
    consumed_capacities = calculate_consumption(problem, path)

    if consumed_capacities[n - 1] + problem.get_capacity(first) > problem.get_capacity(path[0]):
        return None, None, None
    for second in range(0, n):
        before = calculate_distance_between_indices(problem, path[second], path[second + 1])
        after = calculate_distance_between_indices(problem, path[second], first) \
                + calculate_distance_between_indices(problem, first, path[second + 1])
        delta = after - before
        if delta < min_delta:
            min_delta = delta
            label = second

    improved_path_third = path[:(label + 1)] + [first] + path[(label + 1):]
    return improved_path_third, min_delta, label


def do_eject_three_paths(path_first, path_second, path_third, first, second, third):
    return path_first[:first] + [path_third[third]] + path_first[(first + 1):], \
           path_second[:second] + [path_first[first]] + path_second[(second + 1):], \
           path_third[:third] + [path_second[second]] + path_third[(third + 1):]


def eject_three_paths(problem, path_first, path_second, path_third):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    n_third = len(path_third) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)
    consumed_capacities_third = calculate_consumption(problem, path_third)

    for first in range(1, n_first):
        for second in range(1, n_second):
            if consumed_capacities_second[n_second - 1] + problem.get_capacity(path_first[first]) - problem.get_capacity(path_second[second]) > problem.get_capacity(path_second[0]):
                continue
            for third in range(1, n_third):
                if consumed_capacities_third[n_third - 1] + problem.get_capacity(path_second[second]) - problem.get_capacity(path_third[third]) > problem.get_capacity(path_third[0]):
                    continue
                if consumed_capacities_first[n_first - 1] + problem.get_capacity(path_third[third]) - problem.get_capacity(path_first[first]) > problem.get_capacity(path_first[0]):
                    continue
                before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_third[third - 1], path_third[third]) \
                    + calculate_distance_between_indices(problem, path_third[third], path_third[third + 1])
                after = calculate_distance_between_indices(problem, path_first[first - 1], path_third[third]) \
                    + calculate_distance_between_indices(problem, path_third[third], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_third[third - 1], path_second[second]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_third[third + 1])
                delta = after - before
                if delta < min_delta:
                    min_delta = delta
                    label = first, second, third
                    improved_path_first, improved_path_second, improved_path_third = do_eject_three_paths(
                        path_first, path_second, path_third, label[0], label[1], label[2])
                    return improved_path_first, improved_path_second, improved_path_third, min_delta, label
    if label is None:
        return None, None, None, None, None
    else:
        improved_path_first, improved_path_second, improved_path_third = do_eject_three_paths(
            path_first, path_second, path_third, label[0], label[1], label[2])
        return improved_path_first, improved_path_second, improved_path_third, min_delta, label

def  get_exact_lengths_for_exchange_two_paths(action):
    if action in [5, 6, 7]:
        return [action - 4, action - 4]
    elif action in range(12, 25):
        exact_lengths = [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 2],
            [1, 4],
            [4, 1],
            [2, 4],
            [4, 2],
            [3, 4],
            [4, 3],
            [4, 4],
        ]
        return exact_lengths[action - 12]
    else:
        return None

def improve_solution_by_action(step, problem, solution, action):
    improved_solution = copy.deepcopy(solution)
    all_delta = 0.0
    num_paths = len(improved_solution)

    if action in ([1, 2, 3] + list(range(25, 28))):
        for path_index in range(num_paths):
            modified = problem.should_try(action, path_index)
            while modified:
                if action == 1:
                    improved_path, delta, label = two_opt_path(problem, improved_solution[path_index])
                elif action == 2:
                    improved_path, delta, label = exchange_path(problem, improved_solution[path_index])
                else:
                    exact_lengths = {
                        3: 1,
                        4: 2,
                        5: 3,
                        6: 4,
                        25: 2,
                        26: 3,
                        27: 4
                    }
                    improved_path, delta, label = relocate_path(problem, improved_solution[path_index], exact_length=exact_lengths[action])
                if label:
                    modified = True
                    problem.mark_change_at(step, [path_index])
                    improved_solution[path_index] = improved_path
                    all_delta += delta
                else:
                    modified = False
                    problem.mark_no_improvement(step, action, path_index)
        return improved_solution, all_delta

    for path_index_first in range(num_paths - 1):
        for path_index_second in range(path_index_first + 1, num_paths):
            modified = problem.should_try(action, path_index_first, path_index_second)
            if action in ([4, 5, 6, 7] + list(range(12, 25))):
                while modified:
                    if action == 4:
                        improved_path_first, improved_path_second, delta, label = cross_two_paths(
                            problem, improved_solution[path_index_first], improved_solution[path_index_second])
                        if not label:
                            improved_path_first, improved_path_second, delta, label = cross_two_paths(
                                problem, improved_solution[path_index_first], improved_solution[path_index_second][::-1])
                    else:
                        improved_path_first, improved_path_second, delta, label = exchange_two_paths(
                            problem, improved_solution[path_index_first], improved_solution[path_index_second],
                            get_exact_lengths_for_exchange_two_paths(action))
                    if label:
                        modified = True
                        problem.mark_change_at(step, [path_index_first, path_index_second])
                        improved_solution[path_index_first] = improved_path_first
                        improved_solution[path_index_second] = improved_path_second
                        all_delta += delta
                    else:
                        modified = False
                        problem.mark_no_improvement(step, action, path_index_first, path_index_second)

            while action in [8, 9, 10] and modified:
                modified = False
                improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                    problem, improved_solution[path_index_first], improved_solution[path_index_second], action - 7)
                if label:
                    modified = True
                    problem.mark_change_at(step, [path_index_first, path_index_second])
                    improved_solution[path_index_first] = improved_path_first
                    improved_solution[path_index_second] = improved_path_second
                    all_delta += delta
                improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                    problem, improved_solution[path_index_second], improved_solution[path_index_first], action - 7)
                if label:
                    modified = True
                    problem.mark_change_at(step, [path_index_first, path_index_second])
                    improved_solution[path_index_first] = improved_path_second
                    improved_solution[path_index_second] = improved_path_first
                    all_delta += delta
                if not modified:
                    problem.mark_no_improvement(step, action, path_index_first, path_index_second)

            while action == 11 and modified:
                # return improved_solution, all_delta
                modified = False
                improved_path_first, improved_path_second, delta, customer_index = eject_two_paths(
                    problem, improved_solution[path_index_first], improved_solution[path_index_second])
                if customer_index:
                    for path_index_third in range(num_paths):
                        if path_index_third == path_index_first or path_index_third == path_index_second:
                            continue
                        improved_path_third, delta_insert, label = insert_into_path(improved_solution[path_index_third], customer_index, problem)
                        if label is not None and (delta + delta_insert) < -EPSILON:
                            modified = True
                            problem.mark_change_at(step, [path_index_first, path_index_second, path_index_third])
                            improved_solution[path_index_first] = improved_path_first
                            improved_solution[path_index_second] = improved_path_second
                            improved_solution[path_index_third] = improved_path_third
                            all_delta += delta + delta_insert
                            break
                improved_path_first, improved_path_second, delta, customer_index = eject_two_paths(
                    problem, improved_solution[path_index_second], improved_solution[path_index_first])
                if customer_index:
                    for path_index_third in range(num_paths):
                        if path_index_third == path_index_first or path_index_third == path_index_second:
                            continue
                        improved_path_third, delta_insert, label = insert_into_path(improved_solution[path_index_third], customer_index, problem)
                        if label is not None and (delta + delta_insert) < -EPSILON:
                            modified = True
                            problem.mark_change_at(step, [path_index_first, path_index_second, path_index_third])
                            improved_solution[path_index_first] = improved_path_second
                            improved_solution[path_index_second] = improved_path_first
                            improved_solution[path_index_third] = improved_path_third
                            all_delta += delta + delta_insert
                            break
                if not modified:
                    problem.mark_no_improvement(step, action, path_index_first, path_index_second)
    return improved_solution, all_delta