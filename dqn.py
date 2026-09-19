#! python3

import argparse
import collections
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.queue = collections.deque(maxlen=self.memory_size)
        # END STUDENT SOLUTION
        pass

    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        batch = random.choices(self.queue, k=self.batch_size)
        return batch
        # END STUDENT SOLUTION
        pass

    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.queue.append(transition)
        # END STUDENT SOLUTION
        pass


class DeepQNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        double_dqn,
        lr_q_net=2e-4,
        gamma=0.99,
        epsilon=0.05,
        target_update=50, 
        burn_in=10000,
        replay_buffer_size=50000,
        replay_buffer_batch_size=32,
        device="cpu",
    ):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device
        self.num_updates = 0

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size),
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay_buffer = ReplayMemory(
            replay_buffer_size, replay_buffer_batch_size
        )
        self.q_net = q_net_init().to(device)
        self.q_target = q_net_init().to(device)

        self.q_net_network_optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)

        # Target network merely copies the policy and should start from the same random state
        # Also set to eval to not store any gradients in it
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        self.q_target.requires_grad_(False)
        # END STUDENT SOLUTION

    def forward(self, state, action, reward, new_state, dones):
        # Given a minibatch of transitions, return:
        #   q_values: Q(s_j, a_j) under the online network, shape (batch,)
        #   targets:  the TD target y_j, shape (batch,)
        # Use the correct network for the target based on self.double_dqn.
        # BEGIN STUDENT SOLUTION
        # state is (batch, state_dim)
        # self.q_net(state) is (batch, action_dim)
        # q_values is (batch,)

        q_values = self.q_net(state).gather(
            dim=1, index=action.unsqueeze(1)).squeeze(1)

        ## we do not want the target gradients
        with torch.no_grad():
            if self.double_dqn:
                next_action = self.q_net(new_state).argmax(dim = 1)
                next_q_target_values = self.q_target(new_state).gather(
                    dim=1, index=next_action.unsqueeze(1)).squeeze(1)

            else:
                next_q_target_values = self.q_target(new_state).max(dim = 1).values
            targets = reward + (~dones).float() * self.gamma * next_q_target_values
        return q_values, targets
        # END STUDENT SOLUTION

    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION       
        state = torch.as_tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0)
        ## get action_vals from the current online poicy
        with torch.no_grad():
            action_vals = self.q_net(state)


        if stochastic:
            if random.random() < self.epsilon:
                action = random.randrange(self.action_size)
            else:
                action = action_vals.argmax(dim=1).item()
        else:
            action = action_vals.argmax(dim=1).item()

        return action
        # END STUDENT SOLUTION     

    def run(self, env, max_steps, num_episodes, train):
        # Creating this function like the last assignment for simplicity
        total_rewards = []

        # prefilling the buffer
        state, _ = env.reset()
        while train and len(self.replay_buffer.queue) < self.burn_in:
            action = random.randrange(self.action_size)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay_buffer.append([state, action, reward, next_state, done])
            state = next_state
            if terminated or truncated:
                state, _ = env.reset()
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0


            for t in range(max_steps):
                action = self.get_action(state, stochastic=train)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
 
                if train:
                    self.replay_buffer.append(
                        [state, action, reward, next_state, done] # SARST, like SARSA getit? :D
                    )
                    # if len(self.replay_buffer) >= self.burn_in:
                    self.q_net_network_optimizer.zero_grad()

                    states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample_batch())
                    states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
                    actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
                    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                    next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
                    dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

                    q_net_outputs , q_target_values = \
                        self.forward(states, actions, rewards, next_states, dones)
                    
                    loss = nn.functional.mse_loss(q_net_outputs, q_target_values)
                    loss.backward()
                    self.q_net_network_optimizer.step()

                    self.num_updates+=1
                    if self.num_updates % self.target_update == 0:
                        self.q_target.load_state_dict(self.q_net.state_dict())
 
                total_reward += reward
                state = next_state
 
                if done:
                    break
 
            total_rewards.append(total_reward)
 
        return total_rewards # or don't, we don't use it in this assignment


def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f"Starting: {graph_name}")

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION

    D = np.asarray(mean_undiscounted_returns)
    average_total_rewards = D.mean(axis=0)
    min_total_rewards = D.min(axis=0)
    max_total_rewards = D.max(axis=0)

    # plot the total rewards
    os.makedirs("./graphs", exist_ok=True)
    xs = [(i + 1) * test_frequency for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(f"./graphs/{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--test_frequency",
        type=int,
        default=100,
        help="Number of training episodes between test episodes",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of test episodes per checkpoint",
    )
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = "cpu"
    agents = [
        DeepQNetwork(
            state_size,
            action_size,
            double_dqn=args.double_dqn,
            device=device,
        )
        for _ in range(args.num_runs)
    ]

    num_checkpoints = args.num_episodes // args.test_frequency
    mean_undiscounted_returns = np.zeros((args.num_runs, num_checkpoints))

    for run_idx, agent in enumerate(agents):
        print(f"Run {run_idx + 1}/{args.num_runs}")
 
        for checkpoint_idx in range(num_checkpoints):
            agent.run(env, args.max_steps, args.test_frequency, train=True)
            test_rewards = agent.run(env, args.max_steps, args.num_test_episodes, train=False)

            mean_reward = float(np.mean(test_rewards))
            mean_undiscounted_returns[run_idx, checkpoint_idx] = mean_reward
 
            print(
                f"Episode {(checkpoint_idx + 1) * args.test_frequency}: {mean_reward:.2f}"
            )

    graph_name = "DoubleDQN" if args.double_dqn else "DQN"
    graph_agents(
        graph_name,
        mean_undiscounted_returns,
        args.test_frequency,
        args.max_steps,
        args.num_episodes,
    )
    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()
