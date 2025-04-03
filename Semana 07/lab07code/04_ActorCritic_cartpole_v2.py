import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Normalización de estados
class StateNormalizer:
    def __init__(self, state_size):
        self.mean = np.zeros(state_size)
        self.std = np.ones(state_size)
        self.count = 1e-5
    
    def update(self, state):
        self.mean = 0.99 * self.mean + 0.01 * state
        self.std = 0.99 * self.std + 0.01 * (state - self.mean) ** 2
        self.count += 1
    
    def normalize(self, state):
        return (state - self.mean) / (np.sqrt(self.std) + 1e-8)

# Redes mejoradas
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Algoritmo mejorado
def actor_critic(env, actor_net, critic_net, actor_optimizer, critic_optimizer, gamma=0.99, num_episodes=1000, batch_size=5):
    normalizer = StateNormalizer(state_size=env.observation_space.shape[0])
    avg_returns = []
    best_mean_reward = -np.inf
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        total_reward = 0
        
        while not done:
            # Normalizar estado
            normalizer.update(state)
            state = normalizer.normalize(state)
            
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = actor_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            
            next_state = normalizer.normalize(next_state)
            state_value = critic_net(state_tensor)
            
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            
            state = next_state
        
        avg_returns.append(total_reward)
        
        # Cálculo de ventajas con GAE
        values = torch.cat(values).squeeze()
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            gae = delta + gamma * gae
            returns.insert(0, gae + values[t])
        
        returns = torch.tensor(returns)
        advantage = returns - values
        
        # Actualización por lotes
        actor_loss = -torch.cat(log_probs) * advantage.detach()
        critic_loss = nn.functional.smooth_l1_loss(values, returns)
        
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.mean().backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        
        # Guardar el mejor modelo
        if np.mean(avg_returns[-100:]) > best_mean_reward:
            best_mean_reward = np.mean(avg_returns[-100:])
            torch.save(actor_net.state_dict(), "actor_optimized.pth")
            torch.save(critic_net.state_dict(), "critic_optimized.pth")
        
        # Log de progreso
        if (episode + 1) % 10 == 0:
            print(f"Episodio {episode + 1}/{num_episodes}, Retorno: {total_reward:.1f}, Mejor Media: {best_mean_reward:.1f}")
        
        # Parada temprana si resuelve el entorno
        if np.mean(avg_returns[-100:]) >= 195:
            print(f"¡Resuelto en el episodio {episode + 1}!")
            break

# Configuración
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inicializar redes
actor_net = ActorNetwork(state_size, action_size)
critic_net = CriticNetwork(state_size)

# Optimizadores mejorados
actor_optimizer = optim.Adam(actor_net.parameters(), lr=0.0005)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=0.001)

# Entrenamiento
actor_critic(env, actor_net, critic_net, actor_optimizer, critic_optimizer, num_episodes=2000)
