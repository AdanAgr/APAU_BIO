import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# Actor (Policy) Network mejorada
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)  # Regularización
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Solo durante entrenamiento
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Critic (Value) Network mejorada
class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)  # Regularización
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Solo durante entrenamiento
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Algoritmo Actor-Critic optimizado
def actor_critic(env, actor_net, critic_net, actor_optimizer, critic_optimizer, 
                gamma=0.99, num_episodes=1000, save_path=None):
    actor_losses = []  # Para almacenar pérdidas del actor
    critic_losses = []  # Para almacenar pérdidas del crítico
    avg_returns = []  # Para trackear el retorno promedio
    best_mean_reward = -np.inf  # Para guardar el mejor modelo
    
    # Schedulers para ajustar learning rate
    actor_scheduler = optim.lr_scheduler.StepLR(actor_optimizer, step_size=300, gamma=0.9)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=300, gamma=0.9)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_actor_loss = 0
        episode_critic_loss = 0
        total_reward = 0
        steps = 0
        
        while not done:
            # Convertir estado a tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Actor elige acción
            action_probs = actor_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Tomar acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Convertir próximo estado a tensor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            
            # Crítico calcula valores
            value_s = critic_net(state_tensor)
            value_s_prime = critic_net(next_state_tensor).detach()
            
            # Calcular ventaja normalizada
            td_error = reward + gamma * value_s_prime - value_s
            advantage = (td_error - td_error.mean()) / (td_error.std() + 1e-8)
            
            # Pérdida del actor con entropía para exploración
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
            actor_loss = -log_prob * advantage - 0.01 * entropy
            
            # Pérdida del crítico más estable (Huber loss)
            critic_loss = nn.functional.smooth_l1_loss(value_s, reward + gamma * value_s_prime)
            
            # Actualizar redes
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            # Clip de gradientes para estabilidad
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 0.5)
            
            actor_optimizer.step()
            critic_optimizer.step()
            
            # Acumular pérdidas
            episode_actor_loss += actor_loss.item()
            episode_critic_loss += critic_loss.item()
            
            state = next_state
        
        # Ajustar learning rates
        actor_scheduler.step()
        critic_scheduler.step()
        
        # Guardar pérdidas y retorno
        avg_returns.append(total_reward)
        actor_losses.append(episode_actor_loss/steps)
        critic_losses.append(episode_critic_loss/steps)
        
        # Guardar el mejor modelo
        current_mean_reward = np.mean(avg_returns[-100:])
        if current_mean_reward > best_mean_reward:
            best_mean_reward = current_mean_reward
            if save_path:
                torch.save({
                    "actor": actor_net.state_dict(),
                    "critic": critic_net.state_dict(),
                    "avg_return": best_mean_reward
                }, save_path)
        
        # Log de progreso
        if (episode + 1) % 10 == 0:
            print(f"Episodio {episode + 1}/{num_episodes}, Retorno: {total_reward:.1f}, Media (últimos 100): {current_mean_reward:.1f}, Mejor Media: {best_mean_reward:.1f}")
        
        # Parada temprana si resuelve el entorno
        if current_mean_reward >= 195:  # Umbral para CartPole
            print(f"¡Resuelto en el episodio {episode + 1}!")
            break
    
    # Gráficos de resultados
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(actor_losses, label="Pérdida Actor")
    plt.plot(critic_losses, label="Pérdida Crítico", color="green")
    plt.xlabel("Episodio")
    plt.ylabel("Pérdida")
    plt.title("Pérdidas durante el Entrenamiento")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_returns, label="Retorno por Episodio", color="orange")
    plt.xlabel("Episodio")
    plt.ylabel("Retorno")
    plt.title("Retorno durante el Entrenamiento")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Función para ejecutar el entorno en modo visualización
def run_human_mode(env, actor_net):
    state, _ = env.reset()
    env.render()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = actor_net(state_tensor)
        action = torch.argmax(action_probs).item()
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

# Programa Principal
if __name__ == "__main__":
    # Configuración
    train = True  # Cambiar a False para cargar modelo y visualizar
    env_name = "CartPole-v1"  # "FrozenLake-v1" para Frozen Lake
    save_path = f"{env_name}_actor_critic_optimized.pth"

    # Inicializar entorno
    env = gym.make(env_name, render_mode="human" if not train else None)
    state_size = env.observation_space.shape[0] if env_name == "CartPole-v1" else env.observation_space.n
    action_size = env.action_space.n
    
    # Inicializar redes
    actor_net = ActorNetwork(state_size, action_size)
    critic_net = CriticNetwork(state_size)
    
    # Optimizadores
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=0.001, weight_decay=1e-4)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=0.002, weight_decay=1e-4)
    
    if train:
        # Entrenar el modelo
        print("Comenzando entrenamiento...")
        actor_critic(env, actor_net, critic_net, actor_optimizer, critic_optimizer,
                    gamma=0.99, num_episodes=2000, save_path=save_path)
    else:
        # Cargar modelo y visualizar
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            actor_net.load_state_dict(checkpoint["actor"])
            critic_net.load_state_dict(checkpoint["critic"])
            print(f"Modelo cargado de {save_path}")
            print(f"Mejor retorno promedio durante entrenamiento: {checkpoint['avg_return']:.2f}")
            run_human_mode(env, actor_net)
        else:
            print(f"No se encontró modelo guardado en {save_path}")