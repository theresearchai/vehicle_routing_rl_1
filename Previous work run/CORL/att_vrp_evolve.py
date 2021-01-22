import argparse
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from pathlib import Path
import pickle

from utils import load_model
from problems import CVRP

from time import time
from tqdm import tqdm

def fitness(batch, model, params, val=False):
    vector_to_parameters(params, model.parameters())

    model.set_decode_type('greedy')
    model.eval()
    with torch.no_grad():
        length, _ = model(batch)

    return length.mean()

def quick_test(dataset, model, params, batch_size):
    avg = 0
    for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
        avg += (1/(i+1)) * (fitness(batch, model, params) - avg)
        if i == 9:
            return avg

def save(model, params, history, savedir, start, epoch, check=True):
    vector_to_parameters(params, model.parameters())

    if check:
        torch.save(model,'{}/epoch{}-evo-model.pt'.format(savedir,epoch))
    else:
        hr_time = int(round((time()-start)/3600))
        torch.save(model,'{}/{}hr-evo-model.pt'.format(savedir,hr_time))
        with open(f'{savedir}/fitness_history_{hr_time}.pickle', 'wb') as f:
            pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

def train(num_epochs, vis_iter, save_iter, sigma, lr, problem_size,
          batch_size, dataset_size, load_source, save_hrs, savedir):
    # setup
    torch.manual_seed(1234)
    save_hrs.sort()

    # load model
    if load_source is not None:
        model = torch.load(load_source, map_location=torch.device('cpu'))
    else:
        model = torch.load('empty', map_location=torch.device('cpu'))
    params = parameters_to_vector(model.parameters())

    # make dataset
    dataset = CVRP.make_dataset(size=problem_size, num_samples=dataset_size)
    val_dataset = CVRP.make_dataset(size=problem_size, num_samples=batch_size*10)

    # report starting fitness
    start_fitness = quick_test(val_dataset, model, params, batch_size).item()
    print(f'Fitness started at \t{start_fitness}')
    print('-' * 19)
    # training loop
    fitness_history = [[start_fitness, 0]]
    start_time = time()
    npop = 5
    cost = torch.zeros(npop)
    for epoch in range(num_epochs):

        # get cost for each worker
        for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
            eps = torch.randn(npop, len(params))
            for i in range(npop):
                cost[i] = fitness(batch, model, params + sigma * eps[i])

            # update parameters by following cost
            cost = (cost - torch.mean(cost))/torch.std(cost)
            params -= lr/(npop*sigma) * torch.matmul(torch.transpose(eps,0,1),cost)

        # print current fitness occasionally
        if epoch % vis_iter == vis_iter - 1:
            f = quick_test(val_dataset, model, params, batch_size).item()
            print(f'Epoch {epoch + 1}/{num_epochs} \t\t{f}')
            fitness_history.append([f,time() - start_time])

        # save current parameters occasionally
        if epoch % save_iter == save_iter - 1:
            save(model, params, fitness_history, savedir, start_time,epoch)
        for hr in save_hrs:
            if (time() - start_time) > hr*3600:
                save_hrs.remove(hr)
                save(model, params, fitness_history, savedir, start_time,epoch, False)


parser = argparse.ArgumentParser(description='Finetune the trained attention model using OpenAI\'s natural evolution strategy')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--dataset_size', default=128000, type=int)
parser.add_argument('--vis_iter', default=1, type=int)
parser.add_argument('--save_iter', default=10, type=int)
parser.add_argument('--sigma', default=0.01, type=float)
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--problem_size', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--load_source', '-l', default=None, type=str)
parser.add_argument('--save_hrs', nargs='+', type=int)
parser.add_argument('--save_dir', default='../models/att_evo', type=str)
args = parser.parse_args()

train(args.epochs, args.vis_iter, args.save_iter, args.sigma, args.lr, args.problem_size, args.batch_size, args.dataset_size, args.load_source, args.save_hrs, args.save_dir)
