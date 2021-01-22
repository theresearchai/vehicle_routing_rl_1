import argparse

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def get_options():
    parser = argparse.ArgumentParser(description="Meta optimization")
    parser.add_argument('--name', type=str, default='l2i_model', help='Name of model')
    parser.add_argument('--folder', type=str, default='../models/l2i', help='Save folder')
    
    parser.add_argument('--epoch_size', type=int, default=5120000, help='Epoch size')

    parser.add_argument('--num_training_points', type=int, default=100, help="size of the problem for training")
    parser.add_argument('--num_test_points', type=int, default=100, help="size of the problem for testing")
    parser.add_argument('--num_episode', type=int, default=40000, help="number of training episode")
    parser.add_argument('--num_paths_to_ruin', type=int, default=2, help="")
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--max_rollout_steps', type=int, default=20000, help="maximum rollout steps")
    parser.add_argument('--max_rollout_seconds', type=int, default=1000, help="maximum rollout time in seconds")
    parser.add_argument('--detect_negative_cycle', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--epsilon_greedy', type=float, default=0.05, help="")
    parser.add_argument('--sample_actions_in_rollout', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--max_no_improvement', type=int, default=6, help="")
    parser.add_argument('--num_actions', type=int, default=27, help="dimension of action space")
    
    parser.add_argument('--problem_seed', type=int, default=1, help="problem generating seed")
    
    parser.add_argument('--num_embedded_dim', type=int, default=64, help="")
    parser.add_argument('--discount_factor', type=float, default=1.0, help="discount factor of policy network")
    parser.add_argument('--policy_learning_rate', type=float, default=0.001, help="learning rate of policy network")
    parser.add_argument('--hidden_layer_dim', type=int, default=64, help="dimension of hidden layer in policy network")
    
    parser.add_argument('--num_history_action_use', type=int, default=0, help="number of history actions used in the representation of current state")

    parser.add_argument('--training_model', type=str, default=None, help="")
    parser.add_argument('--test_model', type=str, default=None, help="")
    parser.add_argument('--initial_solutions', type=str, default=None, help="path to pickled array of initial solutions")
    parser.add_argument('--max_num_training_epsisodes', type=int, default=10000000, help="")
    
    parser.add_argument('--save_hrs', nargs='+', type=int, default=[1,2,3], help='Array of hours in which the curr model will be saved (int only)')

    config = parser.parse_args()
    config.save_hrs.sort()
    return config
