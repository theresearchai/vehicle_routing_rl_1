## The Vehicle Routing Problem (VRP)

The goal in VRP is to find optimal routes for multiple vehicles visiting a set of locations. When we have a single vehicle, the problem reduces to a traveling salesman problem (TSP). Optimal could either mean minimizing the time, distance, or a combination of both while visiting the locations in the route. Each location is visited only once, by only one vehicle, and each vehicle has a limited capacity. The problem also could have a time constraint where each location must be visited within a certain amount of time.

There are various ways to solve VRP. A naive way is to do a combinatorial search over the search space and choose the path that minimizes the objective (distance or time). However, the problem grows exponentially as the number of nodes and vehicles increase and quickly becomes infeasible to be solved by this naive approach. We will use reinforcement learning (RL) to try to solve VRP since other conventional “unintelligent” approaches such as dynamic programming have struggled to find a solution when the size of the problem gets realistic.

## Data

This section showcases TSP and VRP through visualizations. First, let us look at TSP which is shown in Figure 1. There are 100 nodes (a.k.a. locations) shown as blue dots and one vehicle shown as a red dot. The vehicle moves to the nearest node from a source node and halts when it covers all the nodes.

![][tsp]

Figure 1: A basic model to solve the traveling salesman problem (TSP). Source: Wouter Kool, Herke van Hoof, Max Welling(2019), 'Attention, Learn to Solve Routing Problems!', Retrieved from: https://github.com/wouterkool/attention-learn-to-route

Next, we look at VRP. Figure 2 shows VRP with 100 nodes and 11 vehicles. The left and right side plots in the figure depict two possible sets of routes. The left set of routes has a total distance of 14.06 units which is lower than the distance corresponding to the set of routes shown in the right plot.

![][vrp]

Figure 2: Vehicle routing problem(VRP) with 100 nodes. Source: Wouter Kool, Herke van Hoof, Max Welling(2019), 'Attention, Learn to Solve Routing Problems!', Retrieved from: https://github.com/wouterkool/attention-learn-to-route

For our problem, we are planning to use the same data simulator that generated data for Figure 1. We are going to work with around 1000 - 10000 nodes where each node will have certain properties such as x and y coordinates in the Eucledian space and a vector containing the distance to other nodes. The number of vehicles will be a hyperparameter but we suspect it will most likely be in the range 2-100. The goal of the project will be to minimize the sum of distances covered by all the vehicles with the condition that each node should be visited by at least and at most one vehicle only.

## Self-learning

In the past five weeks, the team has been taking RL classes to solidify the theoretical knowledge in the area. The main areas covered are as follows;

- Markov decision process (MDP) which describes reinforcement learning in an environment that is fully observable.

- Use of dynamic programming if given an MDP to:

	- Plan.

	- Predict the value functions (using the Bellman expectation equation)

	- Control to find the optimal value functions (using Bellman optimality equation).

- Use of model-free techniques (such as Markov chains and temporal-difference learning) if MDP is unknown to:

	- Predict the value function by estimating it.

	- Control by finding the value function using Q-learning which is an interesting concept and also highly relevant to this project.

- Value function approximation using approximators like neural networks with stochastic gradient descent.

	- An area of interest is the use of deep Q-networks (DQN) as a batch approximation method.

- Use policy gradients to get to an optimal policy.

The next steps are to continue learning RL, conduct a literature review, and getting familiar with VRP by generating synthetic data from different distributions.



[tsp]: ./resources/figures/001-tsp.gif "Figure 1: A basic model to solve the traveling salesman problem (TSP). Source: Wouter Kool, Herke van Hoof, Max Welling(2019), 'Attention, Learn to Solve Routing Problems!', Retrieved from: https://github.com/wouterkool/attention-learn-to-route"

[vrp]: ./resources/figures/002-vrp.png "Figure 2: Vehicle routing problem(VRP) with 100 nodes. Source: Wouter Kool, Herke van Hoof, Max Welling(2019), 'Attention, Learn to Solve Routing Problems!', Retrieved from: https://github.com/wouterkool/attention-learn-to-route"
