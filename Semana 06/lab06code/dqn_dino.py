import pygame
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time

pygame.init()

# Global Constants y Configuración de la ventana
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Carga de assets
RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))
BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

# ====================== Clases del Juego Original ======================

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        # Usamos userInput.get(clave, False) para evitar KeyError
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput.get(pygame.K_UP, False) and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput.get(pygame.K_DOWN, False) and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput.get(pygame.K_DOWN, False)):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self, game_speed):
        self.rect.x -= game_speed

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1

# ====================== Entorno Personalizado ======================

class DinoEnv:
    """
    Entorno del juego del dinosaurio que permite realizar pasos (step) y reset,
    similar a un entorno Gym.
    """
    def __init__(self, render=False):
        self.render_mode = render
        self.game_speed = 20
        self.player = Dinosaur()
        self.obstacles = []
        self.cloud = Cloud()
        self.points = 0
        self.clock = pygame.time.Clock()
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.font = pygame.font.Font('freesansbold.ttf', 20)

    def reset(self):
        self.game_speed = 20
        self.player = Dinosaur()
        self.obstacles = []
        self.cloud = Cloud()
        self.points = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        return self.get_state()

    def step(self, action):
        """
        Recibe una acción (0: run, 1: jump, 2: duck) y realiza un update del entorno.
        Devuelve: (next_state, reward, done, info)
        """
        # Simulamos el input del jugador en función de la acción elegida
        userInput = {pygame.K_UP: False, pygame.K_DOWN: False}
        if action == 1:
            userInput[pygame.K_UP] = True
        elif action == 2:
            userInput[pygame.K_DOWN] = True

        self.player.update(userInput)

        # Si no hay obstáculos, se añade uno aleatoriamente
        if len(self.obstacles) == 0:
            r = random.randint(0, 2)
            if r == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif r == 1:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            elif r == 2:
                self.obstacles.append(Bird(BIRD))

        done = False
        reward = 1  # recompensa por cada paso sobrevivido

        # Actualizar cada obstáculo y verificar colisiones
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)
            if self.player.dino_rect.colliderect(obstacle.rect):
                done = True
                reward = -100  # penalización por colisión

        # Eliminar obstáculos que ya han salido de la pantalla
        self.obstacles = [obs for obs in self.obstacles if obs.rect.x > -obs.rect.width]

        # Actualizar fondo y nube
        self.cloud.update(self.game_speed)
        self.x_pos_bg -= self.game_speed
        if self.x_pos_bg <= -BG.get_width():
            self.x_pos_bg = 0

        # Incrementar puntos y ajustar la velocidad del juego
        self.points += 1
        if self.points % 100 == 0:
            self.game_speed += 1

        next_state = self.get_state()
        return next_state, reward, done, {}

    def get_state(self):
        """
        Se construye un vector de estado con 6 características:
        - Posición vertical normalizada del dinosaurio.
        - Velocidad de salto (normalizada).
        - Distancia al obstáculo más cercano (normalizada).
        - Tipo de obstáculo (0: small, 0.5: large, 1: bird).
        - Posición vertical del obstáculo (normalizada).
        - Velocidad del juego (normalizada).
        """
        dino_y = self.player.dino_rect.y / SCREEN_HEIGHT
        dino_jump_vel = self.player.jump_vel / 10.0  # normalización aproximada
        if self.obstacles:
            obs = min(self.obstacles, key=lambda o: o.rect.x)
            obs_dist = (obs.rect.x - self.player.dino_rect.x) / SCREEN_WIDTH
            if isinstance(obs, SmallCactus):
                obs_type = 0.0
            elif isinstance(obs, LargeCactus):
                obs_type = 0.5
            elif isinstance(obs, Bird):
                obs_type = 1.0
            else:
                obs_type = 0.0
            obs_y = obs.rect.y / SCREEN_HEIGHT
        else:
            obs_dist = 1.0
            obs_type = 0.0
            obs_y = 0.0

        game_speed_norm = self.game_speed / 100.0
        state = [dino_y, dino_jump_vel, obs_dist, obs_type, obs_y, game_speed_norm]
        return np.array(state, dtype=np.float32)

    def render(self):
        if self.render_mode:
            SCREEN.fill((255, 255, 255))
            # Dibujar fondo
            SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
            SCREEN.blit(BG, (BG.get_width() + self.x_pos_bg, self.y_pos_bg))
            # Dibujar dinosaurio
            self.player.draw(SCREEN)
            # Dibujar obstáculos
            for obs in self.obstacles:
                obs.draw(SCREEN)
            # Dibujar nube
            self.cloud.draw(SCREEN)
            # Dibujar puntuación
            text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
            SCREEN.blit(text, (1000, 40))
            pygame.display.update()
            self.clock.tick(30)

    def close(self):
        pygame.quit()

# ====================== Agente DQN ======================

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=5e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update_interval=5):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval

        self.memory = deque(maxlen=100000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

        self.episodes_done = 0  # para controlar el update de la red objetivo

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q actual
        current_q = self.model(states).gather(1, actions).squeeze(1)

        # Double DQN
        # Elegimos la acción que maximiza Q en el estado siguiente con la red principal
        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        # Evaluamos esa acción con la red objetivo
        max_next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)

        expected_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping opcional
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Decaimiento epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def end_episode(self):
        """ Llamado al final de cada episodio para actualizar la red objetivo cada cierto intervalo """
        self.episodes_done += 1
        if self.episodes_done % self.target_update_interval == 0:
            self.update_target_network()

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)


# ====================== Bucle Principal de Entrenamiento ======================

def train_dqn(episodes=1000, render=False):
    env = DinoEnv(render=render)
    state_size = 6  # El vector de estado tiene 6 características
    action_size = 3  # Acciones: 0=run, 1=jump, 2=duck
    agent = DQNAgent(state_size, action_size)
    scores = []
    avg_scores = []
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size=64)
            state = next_state
            total_reward += reward
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
        avg_scores.append(avg_score)
        print(f"Episode {e+1}/{episodes} - Score: {total_reward} - Avg Score: {avg_score:.2f} - Epsilon: {agent.epsilon:.2f}")
        
        # Actualizamos la red objetivo cada 10 episodios
        if (e+1) % 10 == 0:
            agent.update_target_network()
    
    # Mostrar gráfico de recompensa promedio
    plt.plot(avg_scores)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DQN Training - Dino Game')
    plt.savefig("training.png", format="png", dpi=300) 
    plt.show()
    env.close()
    agent.save("dqn_dino.pth")

def test_dqn(model_path="dqn_dino.pth", episodes=5):
    # Crear entorno con render activado para ver el juego
    env = DinoEnv(render=True)
    state_size = 6  # El vector de estado tiene 6 características
    action_size = 3  # 0=run, 1=jump, 2=duck

    # Crear agente con la misma arquitectura que usaste en el entrenamiento
    agent = DQNAgent(state_size, action_size)
    # Cargar los pesos entrenados
    agent.load(model_path)
    # Fijar epsilon en 0 para que no haga acciones aleatorias
    agent.epsilon = 0.0

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Renderiza el entorno para visualizar
            env.render()

            # Escoge acción usando la red entrenada
            action = agent.act(state)

            # Paso en el entorno
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Episodio {e+1}/{episodes}, Recompensa total: {total_reward}")

    env.close()


if __name__ == "__main__":

    train = False

    if train:
        train_dqn(episodes=1000, render=False)
    else:
        test_dqn("dqn_dino.pth", episodes=5)

