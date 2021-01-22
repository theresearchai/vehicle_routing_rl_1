import collections
import pickle
import copy
from time import time
import numpy as np
import torch
from utils import *
from options import get_options
from policy import PolicyEstimator
from problem import generate_problem
from perturb import construct_solution
from meta import env_step, should_restart, env_generate_state

def validate_solution(problem, solution, distance=None):
    visited = np.zeros([problem.get_num_customers() + 1])
    for path in solution:
        if path[0] != 0 or path[-1] != 0:
            return False
        consumption = calculate_consumption(problem, path)
        if consumption[-2] > problem.get_capacity(path[0]):
            return False
        for customer in path[1:-1]:
            visited[customer] += 1
    if np.sum(np.ones(problem.get_num_customers()) - visited[1:]) != 0:
        return False
    if distance is not None and np.fabs(distance - calculate_solution_distance(problem, solution)) > EPSILON:
        return False
    return True

def embed_solution_with_attention(problem, solution, config):
    embedded_solution = np.zeros((config.num_training_points, 11))

    for path in solution:
        if len(path) == 2:
            continue
        n = len(path) - 1
        consumption = calculate_consumption(problem, path)
        for index in range(1, n):
            customer = path[index]
            embedded_input = np.zeros([11])
            embedded_input[0] = problem.get_capacity(customer)
            embedded_input[1:3] = problem.get_location(customer)
            embedded_input[3] = problem.get_capacity(0) - consumption[-1]
            embedded_input[4:6] = problem.get_location(path[index - 1])
            embedded_input[6:8] = problem.get_location(path[index + 1])
            embedded_input[8] = problem.get_distance(path[index - 1], customer)
            embedded_input[9] = problem.get_distance(customer, path[index + 1])
            embedded_input[10] = problem.get_distance(path[index - 1], path[index + 1])
            embedded_solution[customer - 1, :] = embedded_input
    return embedded_solution

def run(config):
    torch.manual_seed(config.problem_seed)

    n_ob_space = 5
    trip_emb = 11
    previous_solution = None
    initial_solution = None
    best_solution = None

    distances = []
    solutions = []
    steps = []
    history = []
    start = time()
    
    if config.test_model is not None:
        policy_estimator = torch.load(config.test_model)
    elif config.training_model is not None:
        policy_estimator = torch.load(config.training_model)
    else:
        policy_estimator = PolicyEstimator(config.policy_learning_rate, trip_emb,
                                           config.num_embedded_dim,
                                           config.hidden_layer_dim,
                                           config.num_actions,
                                           n_ob_space)

    if config.initial_solutions is not None:
        initial_solutions = pickle.load(open(config.initial_solutions,'rb'))
        
    Transition = collections.namedtuple("Transition", ["state", "trip", "next_distance", "action", "reward", "next_state", "done"])
    for index_sample in range(config.num_episode):
        states = []
        trips = []
        actions = []
        advantages = []
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
        episode = []
        current_best_distances = []
        current_distances = []

        no_improvement = 0
        for step in range(config.max_rollout_steps):
            action_probs = policy_estimator.predict([state], [embedded_trip])[0].detach()
            action_probs /= torch.sum(action_probs)
            states.append(state)
            trips.append(embedded_trip)
            if (config.test_model is not None and
                    should_restart(min_distance, distance, no_improvement, config)) or no_improvement >= config.max_no_improvement:
                action = 0
                no_improvement = 0
            elif np.random.uniform() < config.epsilon_greedy:
                action = np.random.randint(config.num_actions - 1) + 1
            elif config.sample_actions_in_rollout:
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs.cpu().numpy()) + 1
            else:
                action = np.argmax(action_probs) + 1
            next_state, reward, done, next_solution, next_distance, initial_solution, previous_solution = env_step(step, state, 
                    problem, min_distance, solution, distance, action, env_start_time, config, initial_solution, previous_solution, best_solution)
            if next_distance >= distance - EPSILON:
                no_improvement += 1
            else:
                no_improvement = 0
            current_distances.append(distance)
            current_best_distances.append(min_distance)
            if next_distance < min_distance - EPSILON:
                min_distance = next_distance
                min_step = step
                best_solution = copy.deepcopy(next_solution)
            if done:
                break
            episode.append(Transition(
                state=state, trip=copy.deepcopy(embedded_trip), next_distance=next_distance,
                action=action, reward=reward, next_state=next_state, done=done))
            state = next_state
            solution = next_solution
            embedded_trip = embed_solution_with_attention(problem, solution, config)
            distance = next_distance
        distances.append(min_distance)
        history.append([np.mean(distances),time() - start])
        steps.append(min_step)
        if validate_solution(problem, best_solution, min_distance):
            solutions.append(best_solution)
        else:
            print('invalid solution')
        future_best_distances = np.zeros(len(episode))
        future_best_distances[-1] = episode[-1].next_distance
        step = len(episode) - 2
        while step >= 0:
            if episode[step].action != 0:
                future_best_distances[step] = future_best_distances[step + 1] * config.discount_factor
            else:
                future_best_distances[step] = current_distances[step]
            step = step - 1
        historical_baseline = None
        for t, transition in enumerate(episode):
            if historical_baseline is None:
                if transition.action == 0:
                    historical_baseline = -current_best_distances[t]
                actions.append(0)
                advantages.append(0)
                continue
            if transition.action > 0:
                total_return = -future_best_distances[t]
            else:
                total_return = 0
                actions.append(0)
                advantages.append(0)
                continue
            baseline_value = historical_baseline
            advantage = total_return - baseline_value
            actions.append(transition.action)
            advantages.append(advantage)

        states = np.reshape(np.asarray(states), (-1, n_ob_space)).astype("float32")
        trips = np.reshape(np.asarray(trips), (-1, config.num_training_points, trip_emb)).astype("float32")
        actions = np.reshape(np.asarray(actions), (-1))
        advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")
        if config.test_model is None and index_sample <= config.max_num_training_epsisodes:
            #TODO: figure out what this filtration process is
            filtered_states = []
            filtered_trips = []
            filtered_advantages = []
            filtered_actions = []
            end = 0
            for action_index in range(len(actions)):
                if actions[action_index] > 0:
                    filtered_states.append(states[action_index])
                    filtered_trips.append(trips[action_index])
                    filtered_advantages.append(advantages[action_index])
                    filtered_actions.append(actions[action_index] - 1)
                else:
                    num_bad_steps = config.max_no_improvement
                    end = max(end, len(filtered_states) - num_bad_steps)
                    filtered_states = filtered_states[:end]
                    filtered_trips = filtered_trips[:end]
                    filtered_advantages = filtered_advantages[:end]
                    filtered_actions = filtered_actions[:end]
            filtered_states = filtered_states[:end]
            filtered_trips = filtered_trips[:end]
            filtered_advantages = filtered_advantages[:end]
            filtered_actions = filtered_actions[:end]
            num_states = len(filtered_states)
            if num_states > config.batch_size:
                downsampled_indices = np.random.choice(range(num_states), config.batch_size, replace=False)
                filtered_states = np.asarray(filtered_states)[downsampled_indices]
                filtered_trips = np.asarray(filtered_trips)[downsampled_indices]
                filtered_advantages = np.asarray(filtered_advantages)[downsampled_indices]
                filtered_actions = np.asarray(filtered_actions)[downsampled_indices]
            loss = policy_estimator.update(filtered_states, filtered_trips, filtered_advantages, filtered_actions)
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
        save_path = "./{}/{}_{}.ckpt".format(config.folder,config.name,config.num_episode)
        torch.save(policy_estimator, save_path)
        with open('./{}/distances_{}.pickle'.format(config.folder,config.num_episode), 'wb') as handle:
            pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved in path: %s" % save_path)
    print('solving time = {}'.format(time() - start))
    return distances, solutions, history
                
if __name__ == "__main__":
    run(get_options())
