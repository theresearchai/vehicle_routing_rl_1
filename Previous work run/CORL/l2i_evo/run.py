import collections
import pickle
import copy
from time import time
import numpy as np
import torch
from utils import *
from options import get_options
from policy import PolicyEstimator
from problem import generate_problem, load_problem
from perturb import construct_solution
from meta import env_step, should_restart, env_generate_state

def validate_solution(problem, solution, distance=None):
    visited = [0] * (problem.get_num_customers() + 1)
    for path in solution:
        if path[0] != 0 or path[-1] != 0:
            return False
        consumption = calculate_consumption(problem, path)
        if consumption[-2] > problem.get_capacity(path[0]):
            return False
        for customer in path[1:-1]:
            visited[customer] += 1
    for customer in range(1, len(visited)):
        if visited[customer] != 1:
            return False
    if distance is not None and np.fabs(distance - calculate_solution_distance(problem, solution)) > EPSILON:
        return False
    return True

def embed_solution_with_attention(problem, solution, config):
    embedded_solution = np.zeros((config.num_training_points, config.input_embedded_trip_dim))

    for path in solution:
        if len(path) == 2:
            continue
        n = len(path) - 1
        consumption = calculate_consumption(problem, path)
        for index in range(1, n):
            customer = path[index]
            embedded_input = []
            embedded_input.append(problem.get_capacity(customer))
            embedded_input.extend(problem.get_location(customer))
            embedded_input.append(problem.get_capacity(0) - consumption[-1])
            embedded_input.extend(problem.get_location(path[index - 1]))
            embedded_input.extend(problem.get_location(path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], customer))
            embedded_input.append(problem.get_distance(customer, path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], path[index + 1]))
            for embedded_input_index in range(len(embedded_input)):
                embedded_solution[customer - 1, embedded_input_index] = embedded_input[embedded_input_index]
    return embedded_solution

def run(config):
    torch.manual_seed(config.problem_seed)

    n_ob_space = 5
    previous_solution = None
    initial_solution = None
    best_solution = None

    distances = []
    solutions = []
    history = []
    start = time()
    
    if config.test_model is not None:
        policy_estimator = torch.load(config.test_model)
    elif config.training_model is not None:
        policy_estimator = torch.load(config.training_model)
    else:
        policy_estimator = PolicyEstimator(config.input_embedded_trip_dim,
                                           config.num_embedded_dim,
                                           config.hidden_layer_dim,
                                           config.num_actions,
                                           n_ob_space)

    if config.initial_solutions is not None:
        initial_solutions = pickle.load(open(config.initial_solutions,'rb'))

    params = policy_estimator.get_parameters()
    for index_sample in range(config.num_episode):
        gradient = 0
        n = 0
        sample_start = time()

        problem = generate_problem(config)

        if config.initial_solutions is not None:
            try:
                solution = initial_solutions[index_sample]
            except IndexError:
                config.initial_solutions = None
        else:
            solution = construct_solution(problem, config)
        best_solution = copy.deepcopy(solution)
        embedded_trip = embed_solution_with_attention(problem, solution, config)
        min_distance = calculate_solution_distance(problem, solution)
        min_step = 0
        distance = min_distance

        state = env_generate_state()
        env_start_time = time()

        no_improvement = 0
        for step in range(config.max_rollout_steps):
            if (config.test_model is not None and
                    should_restart(min_distance, distance, no_improvement, config)) or no_improvement >= config.max_no_improvement:
                action = 0
                no_improvement = 0
            else:
            	n += 1
            	eps = torch.randn_like(params)
            	test_params = params + config.sigma*eps
            	action_probs = policy_estimator.predict([state], [embedded_trip], test_params)[0].detach()
            	action_probs /= torch.sum(action_probs)
            	if config.sample_actions_in_rollout:
            	    action = np.random.choice(np.arange(len(action_probs)), p=action_probs.cpu().numpy()) + 1
            	else:
                   action = np.argmax(action_probs) + 1
            next_state, reward, done, next_solution, next_distance, initial_solution, previous_solution = env_step(step, state, 
                    problem, min_distance, solution, distance, action, env_start_time, config, initial_solution, previous_solution, best_solution)
            if next_distance >= distance - EPSILON:
                no_improvement += 1
            else:
                no_improvement = 0
            if next_distance < min_distance - EPSILON:
                min_distance = next_distance
                min_step = step
                best_solution = copy.deepcopy(next_solution)
            if action != 0:
            	gradient += (1/n) * (reward * eps - gradient) 
            if done:
                break
            state = next_state
            solution = next_solution
            embedded_trip = embed_solution_with_attention(problem, solution, config)
            distance = next_distance
        distances.append(min_distance)
        history.append([np.mean(distances),time() - start])
        params -= (config.alpha/config.sigma)*gradient
        policy_estimator.update(params)
        if validate_solution(problem, best_solution, min_distance):
            solutions.append(best_solution)
        else:
            print('invalid solution')
        actual_time = (time() - sample_start)
        if distances:
            print('Sample #{} avg min dist: {} time for samp: {}'.format(index_sample, np.mean(distances),
                                                                         actual_time))
        if config.test_model is None:
            for hr in config.save_hrs:
                if (time() - start) > hr*3600:
                    config.save_hrs.remove(hr)
                    hr_time = int(round((time()-start)/3600))
                    save_path = "./{}/{}_{}hr.ckpt".format(config.folder,config.name, hr_time)
                    torch.save(policy_estimator, save_path)
                    with open('./{}/{}_{}hr.pickle'.format(config.folder,config.name,hr_time), 'wb') as handle:
                        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Model saved in path: %s" % save_path)

    if config.test_model is None:
        save_path = torch.save(policy_estimator, "./{}/{}_{}.ckpt".format(config.folder,config.name,config.num_episode))
        with open('./{}/distances_{}.pickle'.format(config.folder,config.num_episode), 'wb') as handle:
            pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved in path: %s" % save_path)
    print('solving time = {}'.format(time() - start))
    return distances, solutions, history
                
if __name__ == "__main__":
    run(get_options())
