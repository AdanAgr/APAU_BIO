import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# -----------------
# Policy Network
# -----------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# -----------------
# Value Network
# -----------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# -----------------
# Flatten/Unflatten Helpers
# -----------------
def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, new_params):
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(new_params[idx:idx+size].view(p.size()))
        idx += size


# -----------------
# Conjugate Gradient
# -----------------
def conjugate_gradient(Ax, b, max_iter=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = Ax(p)
        alpha = rsold / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# -----------------
# Surrogate Loss
# -----------------
def surrogate_loss(policy, states, actions, advantages, old_log_probs):
    new_probs = policy(states).clamp(min=1e-8)
    new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
    return (advantages * torch.exp(new_log_probs - old_log_probs)).mean()


# -----------------
# KL Divergence
# -----------------
def kl_divergence(old_probs, new_probs):
    return (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=1).mean()


# -----------------
# TRPO Training
# -----------------
def train_trpo(env, policy, value_net, max_episodes=500, gamma=0.99, lam=0.95, max_kl=0.01, save_path="trpo_policy.pth"):
    optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    all_returns = []

    for episode in range(max_episodes):
        obs, info = env.reset()
        done = False

        states = []
        actions = []
        rewards = []

        while not done:
            obs_np = np.array(obs, dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)

            with torch.no_grad():
                probs = policy(obs_tensor).clamp(min=1e-8)
            dist = Categorical(probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs_np)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        states_t = torch.from_numpy(np.array(states, dtype=np.float32))
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64))
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32))

        returns = []
        G = 0
        for r in reversed(rewards_t):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        values = value_net(states_t)
        values_next = torch.cat([values[1:], torch.tensor([0.0])])
        deltas = rewards_t + gamma * values_next - values

        advantages_list = []
        A = 0
        for delta in reversed(deltas):
            A = delta + gamma * lam * A
            advantages_list.insert(0, A)
        advantages_t = torch.tensor(advantages_list, dtype=torch.float32)

        with torch.no_grad():
            old_probs = policy(states_t).clamp(min=1e-8)
            old_log_probs = torch.log(old_probs.gather(1, actions_t.unsqueeze(-1)).squeeze(-1))

        surr = surrogate_loss(policy, states_t, actions_t, advantages_t, old_log_probs)
        grad = torch.autograd.grad(surr, policy.parameters(), retain_graph=True)
        grad = torch.cat([g.view(-1) for g in grad])

        def fisher_vector_product(v):
            new_probs = policy(states_t).clamp(min=1e-8)
            kl = kl_divergence(old_probs, new_probs)
            grads_kl = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
            flat_grads_kl = torch.cat([g.view(-1) for g in grads_kl])

            grad_v = torch.dot(flat_grads_kl, v)
            grads_v = torch.autograd.grad(grad_v, policy.parameters())
            return torch.cat([g.contiguous().view(-1) for g in grads_v])

        step_dir = conjugate_gradient(fisher_vector_product, grad)

        fvp_step_dir = fisher_vector_product(step_dir)
        denom = 0.5 * torch.dot(step_dir, fvp_step_dir)
        if denom.item() < 1e-8:
            print("Warning: denominator in TRPO update is tiny. Skipping update.")
            continue
        step_scale = torch.sqrt((2.0 * max_kl) / denom)
        step_dir = step_dir * step_scale

        old_params = flat_params(policy)
        alpha = 1.0
        for _ in range(10):
            new_params = old_params + alpha * step_dir
            set_params(policy, new_params)

            with torch.no_grad():
                new_probs = policy(states_t).clamp(min=1e-8)
                actual_kl = kl_divergence(old_probs, new_probs)
                new_surr = surrogate_loss(policy, states_t, actions_t, advantages_t, old_log_probs)

            if torch.isnan(new_probs).any() or torch.isnan(actual_kl) or torch.isnan(new_surr):
                alpha *= 0.5
                continue

            if actual_kl > max_kl:
                alpha *= 0.5
            else:
                if new_surr > 0.0:
                    break
                else:
                    alpha *= 0.5
        else:
            print("Line search failed; reverting to old parameters.")
            set_params(policy, old_params)

        value_loss = (returns_t - value_net(states_t)).pow(2).mean()
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        total_return = returns_t.sum().item()
        all_returns.append(total_return)
        print(f"Episode {episode+1}/{max_episodes}, Return: {total_return:.2f}")

    torch.save({"policy": policy.state_dict(), "value_net": value_net.state_dict()}, save_path)
    print(f"Models saved to {save_path}")

    plt.plot(all_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TRPO on CartPole-v1 (Gymnasium)")
    plt.show()


def evaluate_policy(env, policy_path, episodes=10):
    checkpoint = torch.load(policy_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    total_returns = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_return = 0

        while not done:
            obs_np = np.array(obs, dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)

            with torch.no_grad():
                probs = policy(obs_tensor).clamp(min=1e-8)
            dist = Categorical(probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_return += reward
            obs = next_obs

        total_returns.append(total_return)
        print(f"Test Episode {episode+1}/{episodes}, Return: {total_return:.2f}")

    avg_return = np.mean(total_returns)
    print(f"Average Test Return: {avg_return:.2f}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    train_trpo(env, policy, value_net, max_episodes=500, max_kl=0.01)

    evaluate_policy(env, "trpo_policy.pth", episodes=10)