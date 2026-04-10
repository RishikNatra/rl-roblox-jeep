import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import numpy as np
import os
import pandas as pd
import time

if not os.path.exists('plots'):
    os.makedirs('plots')

app = Flask(__name__)

# --- CONFIG ---
STATE_DIM = 9   # Added 1 for the 'Ground Detector' sensor
ACTION_DIM = 4  
LR = 0.0003
UPDATE_INTERVAL = 128 
SUCCESS_REWARD = 1000.0 
CRASH_PENALTY = -500.0 # High penalty for hitting walls/trees/river
STAGNATION_THRESHOLD = 60 
TIME_LIMIT = 120 

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = self.fc(state)
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

device = torch.device("cpu")
model = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()

# Buffers
states_mem, actions_mem, rewards_mem, log_probs_mem, values_mem = [], [], [], [], []
episode_history = []
current_episode_reward = 0
total_steps = 0
last_dist = 9999
episode_start_time = time.time()
no_progress_steps = 0

def plot_rewards():
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_history, color='tab:red', alpha=0.4)
        if len(episode_history) > 5:
            avg = pd.Series(episode_history).rolling(window=10).mean()
            plt.plot(avg, color='black', linewidth=2)
        plt.title("Obstacle Avoidance Training")
        plt.savefig('plots/training_progress.png')
        plt.close('all')
    except: pass

def train():
    global states_mem, actions_mem, rewards_mem, log_probs_mem, values_mem
    if len(states_mem) < 10: return
    S = torch.stack(states_mem)
    A = torch.tensor(actions_mem).unsqueeze(1)
    R = torch.tensor(rewards_mem).unsqueeze(1)
    LP = torch.stack(log_probs_mem).detach()
    V = torch.stack(values_mem).detach()
    R_norm = (R - R.mean()) / (R.std() + 1e-5)
    advantages = (R_norm - V).detach()
    for _ in range(4):
        probs, val = model(S)
        dist = torch.distributions.Categorical(probs)
        new_lp = dist.log_prob(A.squeeze()).unsqueeze(1)
        ratio = torch.exp(new_lp - LP)
        surr1, surr2 = ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss, value_loss = -torch.min(surr1, surr2).mean(), mse_loss(val, R_norm)
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()
    states_mem, actions_mem, rewards_mem, log_probs_mem, values_mem = [], [], [], [], []

@app.route('/act', methods=['POST'])
def act():
    global current_episode_reward, total_steps, last_dist, episode_start_time, no_progress_steps
    try:
        data = request.json
        d, a, s = data['distance'], data['angle'], data['speed']
        is_collision = data.get('collision', False)
        
        # 1. Termination Checks
        elapsed = time.time() - episode_start_time
        progress = last_dist - d
        if progress < 0.1: no_progress_steps += 1
        else: no_progress_steps = 0
        
        is_success = d < 12
        is_failed = is_collision or (elapsed > TIME_LIMIT) or (no_progress_steps > STAGNATION_THRESHOLD)
        
        # 2. Reward Shaping
        step_reward = (a * 6.0) + (progress * 20.0) + (s * 0.2) - 1.5 
        if is_success: step_reward += SUCCESS_REWARD
        if is_collision: step_reward += CRASH_PENALTY # Massive hit for trees/river/walls
            
        current_episode_reward += step_reward
        last_dist = d

        # 3. Decision
        state_t = torch.FloatTensor([*data['sensors'], d/100, a, s/50]).to(device)
        probs, val = model(state_t)
        action = torch.distributions.Categorical(probs).sample()

        states_mem.append(state_t)
        actions_mem.append(action.item())
        rewards_mem.append(step_reward)
        log_probs_mem.append(torch.distributions.Categorical(probs).log_prob(action))
        values_mem.append(val)

        if is_success or is_failed:
            episode_history.append(current_episode_reward)
            current_episode_reward, last_dist, no_progress_steps = 0, 9999, 0
            episode_start_time = time.time()
            plot_rewards()

        total_steps += 1
        if total_steps % UPDATE_INTERVAL == 0: train()
        return jsonify({"action": action.item(), "reset": (is_success or is_failed)})
    except:
        return jsonify({"action": 3, "reset": False})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)