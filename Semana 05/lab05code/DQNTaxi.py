import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Linear output for Q-values

class Agent:
    def __init__(self, env_string, batch_size=64, render_mode=None, update_target_steps=100):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_string, render_mode=render_mode)
        self.input_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
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

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.env.action_space.sample()
        else:
            state_one_hot = np.zeros(self.input_size)
            state_one_hot[state] = 1
            state_tensor = torch.FloatTensor(state_one_hot).to(self.device)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_one_hot = np.zeros((self.batch_size, self.input_size))
        next_states_one_hot = np.zeros((self.batch_size, self.input_size))
        for i, (s, ns) in enumerate(zip(states, next_states)):
            states_one_hot[i][s] = 1
            next_states_one_hot[i][ns] = 1

        states_tensor = torch.FloatTensor(states_one_hot).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_one_hot).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states_tensor).max(1)[0]
        expected_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_steps == 0:
            self.update_target_network()

        return loss.item()

    def train_model(self, epochs=5000, threshold=9.7):
        scores = deque(maxlen=100)

        for epoch in range(epochs):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            scores.append(total_reward)
            mean_score = np.mean(scores)

            if len(self.memory) >= self.batch_size:
                self.replay()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if mean_score >= threshold and len(scores) == 100:
                print(f'Solved after {epoch - 100} episodes ✔')
                torch.save(self.model.state_dict(), 'dqn_taxi.pth')
                return

            if epoch % 100 == 0:
                print(f'[Episode {epoch}] - Mean reward over last 100 episodes: {mean_score:.2f}')

        print('Training complete but not solved. Saving weights...')
        torch.save(self.model.state_dict(), 'dqn_taxi.pth')

    def load_weights_and_visualize(self):
        self.model.load_state_dict(torch.load('dqn_taxi.pth'))
        self.model.eval()

        for episode in range(5):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                state_one_hot = np.zeros(self.input_size)
                state_one_hot[state] = 1
                state_tensor = torch.FloatTensor(state_one_hot).to(self.device)

                with torch.no_grad():
                    action_values = self.model(state_tensor)
                action = torch.argmax(action_values).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            print(f'Episode {episode + 1}: Total reward = {total_reward}')

# Usage
train_mode = False

if train_mode:
    agent = Agent('Taxi-v3', render_mode=None)
    agent.train_model()
else:
    agent = Agent('Taxi-v3', render_mode='human')
    agent.load_weights_and_visualize()
