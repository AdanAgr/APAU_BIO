#!/usr/bin/env python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time

class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output scaled for continuous action space


class Agent():
    def __init__(self, env_string, batch_size=64, render_mode=None, update_target_steps=100):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string, render_mode=render_mode)
        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]  # For continuous action space
        self.action_high = torch.FloatTensor(self.env.action_space.high).to("cpu")
        self.action_low = torch.FloatTensor(self.env.action_space.low).to("cpu")

        self.batch_size = batch_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_target_steps = update_target_steps

        self.model = DQN(self.input_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.target_model = DQN(self.input_size, self.action_size).to(self.device)
        self.update_target_network()

        self.steps_done = 0
        self.scores = []
        self.avg_scores = []
        self.setup_plot()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Mean Reward')
        plt.show()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.avg_scores)
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Mean Reward')
        self.ax.set_title('Mean Reward during Training')
        plt.draw()
        plt.pause(0.001)

    def preprocess_state(self, state):
        if isinstance(state, tuple):
            state = state[0]
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.env.action_space.sample()  # Random continuous action
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.model(state).squeeze(0)
            return torch.clip(action, self.action_low, self.action_high).cpu().numpy()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q-values predicted by the model
        predicted_actions = self.model(states)

        # Calculate loss between predicted and actual actions
        # Note: This assumes actions are treated as continuous values
        loss = nn.MSELoss()(predicted_actions, actions)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_steps == 0:
            self.update_target_network()

    def train_model(self, epochs=1000, threshold=300):
        scores = deque(maxlen=100)

        for epoch in range(epochs):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            scores.append(total_reward)
            mean_score = np.mean(scores)
            self.avg_scores.append(mean_score)

            if len(self.memory) >= self.batch_size:
                self.replay()

            self.update_plot()

            if mean_score >= threshold and len(scores) == 100:
                print(f"Environment solved in {epoch} episodes! Mean Reward: {mean_score}")
                torch.save(self.model.state_dict(), 'dqn_bipedalwalker.pth')
                return

            if epoch % 100 == 0:
                print(f"Episode {epoch}, Mean Reward: {mean_score}")

        print("Training complete. Threshold not reached.")
        torch.save(self.model.state_dict(), 'dqn_bipedalwalker.pth')

    def load_weights_and_visualize(self):
        self.model.load_state_dict(torch.load('dqn_bipedalwalker.pth'))
        self.model.eval()

        for _ in range(5):
            state = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            while not done:
                self.env.render()
                time.sleep(0.05)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.model(state_tensor).squeeze(0).cpu().numpy()
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = self.preprocess_state(state)
        self.env.close()

# Usage
train_mode = False

if train_mode:
    agent = Agent('BipedalWalker-v3')
    agent.train_model()
else:
    agent = Agent('BipedalWalker-v3', render_mode='human')
    agent.load_weights_and_visualize()
