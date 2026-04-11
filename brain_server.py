import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import numpy as np
import os
import time
import signal
import sys

if not os.path.exists('plots'): os.makedirs('plots')

app = Flask(__name__)

# --- CONFIG ---
STATE_DIM = 10  
ACTION_DIM = 4  
LR = 0.00025     
GAMMA = 0.99    
GAE_LAMBDA = 0.95 
PPO_CLIP = 0.2    
ENTROPY_COEFF = 0.02 
UPDATE_INTERVAL = 1024 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "jeep_model.pth") 

class PPOBrain(nn.Module):
    def __init__(self):
        super(PPOBrain, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, ACTION_DIM)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.fc(x)
        return torch.softmax(self.actor(features), dim=-1), self.critic(features)

device = torch.device("cpu")
model = PPOBrain().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()

def save_checkpoint(steps):
    checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'total_steps': steps, 'state_dim': STATE_DIM}
    torch.save(checkpoint, MODEL_PATH)
    print(f"✅ Brain Saved at step {steps}")

def load_checkpoint():
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH)
            if checkpoint.get('state_dim', 0) == STATE_DIM:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("🚀 Brain Loaded Successfully")
                return checkpoint.get('total_steps', 0)
            else:
                print("⚠️ Dimension mismatch in file. Starting fresh.")
        except: print("⚠️ Could not load. Starting fresh.")
    return 0

def signal_handler(sig, frame):
    save_checkpoint(total_steps)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
total_steps = load_checkpoint()

states, actions, rewards, log_probs, is_terminals, values = [], [], [], [], [], []
last_dist, current_ep_reward = 0, 0
last_move_time = time.time()
episode_history = []

def train():
    global states, actions, rewards, log_probs, is_terminals, values
    if len(states) < UPDATE_INTERVAL: return
    S = torch.stack(states).to(device)
    A = torch.tensor(actions).to(device).unsqueeze(1)
    LP = torch.stack(log_probs).detach().to(device)
    V = torch.stack(values).detach().to(device) 
    R = torch.tensor(rewards).to(device).float()
    Mask = torch.tensor([0 if m else 1 for m in is_terminals]).to(device)
    returns, adv, last_gae = torch.zeros_like(R), torch.zeros_like(R), 0
    for t in reversed(range(len(R))):
        next_val = 0 if t == len(R) - 1 else V[t+1]
        delta = R[t] + GAMMA * next_val * Mask[t] - V[t]
        adv[t] = last_gae = delta + GAMMA * GAE_LAMBDA * Mask[t] * last_gae
        returns[t] = adv[t] + V[t]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(5):
        probs, v_curr = model(S)
        dist = torch.distributions.Categorical(probs)
        new_lp = dist.log_prob(A.squeeze()).unsqueeze(1)
        ratio = torch.exp(new_lp - LP)
        surr1 = ratio * adv.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv.unsqueeze(1)
        loss = -torch.min(surr1, surr2).mean() + 0.5 * mse_loss(v_curr, returns.unsqueeze(1)) - ENTROPY_COEFF * dist.entropy().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    states[:], actions[:], rewards[:], log_probs[:], is_terminals[:], values[:] = [], [], [], [], [], []
    save_checkpoint(total_steps)

@app.route('/act', methods=['POST'])
def act():
    global last_dist, total_steps, current_ep_reward, last_move_time
    try:
        data = request.json
        sensors = data.get('sensors', [60]*6)
        # Handle cases where sensors might be missing
        while len(sensors) < 6: sensors.append(0)
        
        d, a, s = data['distance'], data['angle'], data['speed']
        lvl, is_collision = data['level'], data.get('collision', False)

        current_time = time.time()
        if s > 2.0: last_move_time = current_time
        is_stuck = (current_time - last_move_time) > 15.0

        if last_dist == 0: last_dist = d
        progress = (last_dist - d)
        
        reward = -0.01 + (progress * 0.6) + (a * 0.15) 

        done = False
        if d < 15: 
            reward += 15.0
            done = True
            print(f"✨ Level {lvl} Success!")
        elif is_collision or sensors[5] < 3.0 or is_stuck: 
            reward -= 10.0
            done = True
            print(f"💥 Failure at Level {lvl}")

        # Construct 10D State: 6 sensors + dist + angle + speed + level
        state_list = [*([v/60.0 for v in sensors[:6]]), d/1500.0, a, s/100.0, lvl/20.0]
        state_t = torch.FloatTensor(state_list).to(device)
        
        with torch.no_grad():
            probs, val = model(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        states.append(state_t); actions.append(action.item()); rewards.append(reward)
        log_probs.append(dist.log_prob(action)); is_terminals.append(done); values.append(val.squeeze()) 

        current_ep_reward += reward
        last_dist = 0 if done else d
        total_steps += 1
        
        if done:
            episode_history.append(current_ep_reward)
            current_ep_reward = 0
            last_move_time = time.time()
            if len(episode_history) % 10 == 0:
                plt.figure(); plt.plot(episode_history); plt.savefig('plots/progress.png'); plt.close()

        if len(states) >= UPDATE_INTERVAL: train()
        return jsonify({"action": action.item(), "reset": done})
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return jsonify({"action": 3, "reset": True})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=False)