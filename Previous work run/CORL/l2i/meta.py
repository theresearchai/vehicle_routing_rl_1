from utils import calculate_solution_distance, EPSILON
from time import time
from improve import improve_solution_by_action
from perturb import construct_solution
import numpy as np

def env_step(step, state, problem, min_distance, solution, distance, action, env_start_time, config,
        initial_solution, previous_solution, best_solution):
    next_trip, next_distance, initial_solution, previous_solution = env_act(step, problem, min_distance, solution,
            distance, action, initial_solution, previous_solution, best_solution, config)
    next_state = env_generate_state(min_distance, state, action, distance, next_distance)
    reward = distance - next_distance
    done = (time() - env_start_time) >= config.max_rollout_seconds
    return next_state, reward, done, next_trip, next_distance, initial_solution, previous_solution

def env_act(step, problem, min_distance, solution, distance, action, initial_solution, previous_solution,
            best_solution, config):
    if action > 0:
        next_solution, delta = improve_solution_by_action(step, problem, solution, action)
        next_distance = distance + delta
    else:
        problem.record_solution(solution, distance)
        if distance / min_distance < 1.01:
            previous_solution = solution
            next_solution = construct_solution(problem, config, solution, step)
        else:
            previous_solution = best_solution
            next_solution = construct_solution(problem, config, best_solution, step)
        next_distance = calculate_solution_distance(problem, next_solution)
        initial_solution = next_solution
    return next_solution, next_distance, initial_solution, previous_solution

def should_restart(min_distance, distance, no_improvement, config):
    if no_improvement >= config.max_no_improvement:
        return True
    if no_improvement <= config.max_no_improvement - 1:
        return False
    percentage_over = round((distance / min_distance - 1.0) * 100)
    upper_limits = [20, 10, 5, 2]
    return percentage_over >= upper_limits[no_improvement - 2]

def env_generate_state(min_distance=None, state=None, action=None, distance=None, next_distance=None):
    next_state = np.zeros([5])
    if state is not None and action != 0:
        delta = next_distance - distance
        if delta < -EPSILON:
            delta_sign = -1.0
        else:
            delta_sign = 1.0
        next_state[1:] = [next_distance - min_distance, delta, delta_sign, action]
    return next_state
