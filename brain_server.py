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
import threading

if not os.path.exists('plots'): os.makedirs('plots')

app = Flask(__name__)

# --- CONFIG ---
STATE_DIM = 11  
ACTION_DIM = 4  
LR = 0.0005     
GAMMA = 0.99    
GAE_LAMBDA = 0.95 
PPO_CLIP = 0.15    
ENTROPY_COEFF = 0.1 
UPDATE_INTERVAL = 512 
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

brain_lock = threading.Lock()
env_states = {}

states, actions, rewards, log_probs, is_terminals, values = [], [], [], [], [], []
episode_history = []
success_history = []
distance_history = []
entropy_history = []
kl_history = []
critic_loss_history = []
best_success_rate = 0.0

def train():
    global states, actions, rewards, log_probs, is_terminals, values, ENTROPY_COEFF
    if len(states) < UPDATE_INTERVAL: return
    S = torch.stack(states).to(device)
    A = torch.tensor(actions).to(device).unsqueeze(1)
    LP = torch.stack(log_probs).detach().to(device)
    V = torch.stack(values).detach().to(device) 
    R = torch.tensor(rewards).to(device).float()
    
    # Reward Normalization
    R = (R - R.mean()) / (R.std() + 1e-8)
    
    Mask = torch.tensor([0 if m else 1 for m in is_terminals]).to(device)
    returns, adv, last_gae = torch.zeros_like(R), torch.zeros_like(R), 0
    for t in reversed(range(len(R))):
        next_val = 0 if t == len(R) - 1 else V[t+1]
        delta = R[t] + GAMMA * next_val * Mask[t] - V[t]
        adv[t] = last_gae = delta + GAMMA * GAE_LAMBDA * Mask[t] * last_gae
        returns[t] = adv[t] + V[t]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    epoch_kls = []
    epoch_entropies = []
    epoch_critic_loss = []
    
    for _ in range(8):
        probs, v_curr = model(S)
        dist = torch.distributions.Categorical(probs)
        new_lp = dist.log_prob(A.squeeze()).unsqueeze(1)
        ratio = torch.exp(new_lp - LP)
        
        with torch.no_grad():
            log_ratio = new_lp - LP
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
        
        surr1 = ratio * adv.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv.unsqueeze(1)
        
        v_loss = 0.5 * mse_loss(v_curr, returns.unsqueeze(1))
        entropy = dist.entropy().mean()
        loss = -torch.min(surr1, surr2).mean() + v_loss - ENTROPY_COEFF * entropy + 0.01 * approx_kl
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        epoch_kls.append(approx_kl.item())
        epoch_entropies.append(entropy.item())
        epoch_critic_loss.append(v_loss.item())

    kl_history.append(np.mean(epoch_kls))
    entropy_history.append(np.mean(epoch_entropies))
    critic_loss_history.append(np.mean(epoch_critic_loss))
    
    # Entropy decay
    ENTROPY_COEFF = max(0.01, ENTROPY_COEFF * 0.995)
    
    states[:], actions[:], rewards[:], log_probs[:], is_terminals[:], values[:] = [], [], [], [], [], []
    
    # Save checkpoint every 5000+ steps or update
    if total_steps > 0 and total_steps % 5000 < UPDATE_INTERVAL:
        save_checkpoint(total_steps)

@app.route('/act', methods=['POST'])
def act():
    global total_steps, best_success_rate
    try:
        data = request.json
        env_id = data.get('id', 'jeep_main')
        
        with brain_lock:
            if env_id not in env_states:
                env_states[env_id] = {
                    'last_dist': 0, 
                    'current_ep_reward': 0, 
                    'last_move_time': time.time(),
                    'start_time': time.time(),
                    'off_path_since': None,
                    'flipped_since': None
                }
            
            sensors = data.get('sensors', [])
            while len(sensors) < 6: sensors.append(60.0)
            
            d, a, s = data['distance'], data['angle'], data['speed']
            lvl, is_collision = data['level'], data.get('collision', False)
            g_dist = data.get('guideDistance', 0)

            current_time = time.time()
            e_state = env_states[env_id]
            
            if s > 2.0: e_state['last_move_time'] = current_time
            is_stuck = (current_time - e_state['last_move_time']) > 10.0

            if e_state['last_dist'] == 0: e_state['last_dist'] = d
            progress = max(0, e_state['last_dist'] - d)
            
            # Timeout (1 min)
            is_timeout = (current_time - e_state['start_time']) > 60.0
            
            # Off path tracking (> 10s)
            if g_dist > 10.0:
                if e_state['off_path_since'] is None:
                    e_state['off_path_since'] = current_time
            else:
                e_state['off_path_since'] = None
            is_off_path_long = e_state['off_path_since'] is not None and (current_time - e_state['off_path_since']) > 10.0
            
            # Flipped tracking (> 10s)
            is_flipped = data.get('flipped', False)
            if is_flipped:
                if e_state['flipped_since'] is None:
                    e_state['flipped_since'] = current_time
            else:
                e_state['flipped_since'] = None
            is_flipped_long = e_state['flipped_since'] is not None and (current_time - e_state['flipped_since']) > 10.0
            
            # --- Better Reward Shaping ---
            direction_reward = a if a > 0 else a * 2.0 
            speed_efficiency = (s / 100.0) * a 
            
            path_penalty = 0
            if g_dist > 5.0:
                path_penalty = (g_dist - 5.0) * 0.05
                
            reward = -0.05 + (progress * 1.5) + (direction_reward * 0.5) + (speed_efficiency * 1.0) - path_penalty
            
            if e_state['last_dist'] > 0:
                reward += (e_state['last_dist'] - d) * 0.8
                
            if is_stuck or s < 2.0:
                reward -= 2.0

            done = False
            is_success = False
            if d < 15: 
                reward += 30.0
                done = True
                is_success = True
                print(f"✨ [{env_id}] Level {lvl} Success!")
            elif is_off_path_long:
                reward -= 15.0
                done = True
                print(f"🚫 [{env_id}] Resetting... (Off-Path > 10s)")
            elif is_flipped_long:
                reward -= 15.0
                done = True
                print(f"💥 [{env_id}] Resetting... (Flipped > 10s)")
            elif is_timeout:
                reward -= 15.0
                done = True
                print(f"⏰ [{env_id}] Timeout - Over 1 minute")
            elif is_collision or sensors[5] < 3.0 or is_stuck: 
                reward -= 15.0
                done = True
                
                # Help debugging if it's sensor-based, physical collision, or stuck
                if is_stuck:
                    stuck_text = "Stuck"
                elif is_collision:
                    stuck_text = "Collision"
                else:
                    stuck_text = f"Sensor[5] hit <3.0 ({sensors[5]:.1f})"
                    
                print(f"💥 [{env_id}] Failure ({stuck_text}) at Level {lvl} (Dist: {d:.1f})")

            # Construct 11D State: 6 sensors + dist + g_dist + angle + speed + level
            state_list = [
                *([v/60.0 for v in sensors[:6]]), 
                d/1500.0, 
                g_dist/1000.0,
                a, 
                s/100.0, 
                lvl/20.0
            ]
            state_t = torch.FloatTensor(state_list).to(device)
            
            with torch.no_grad():
                probs, val = model(state_t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            states.append(state_t); actions.append(action.item()); rewards.append(reward)
            log_probs.append(dist.log_prob(action)); is_terminals.append(done); values.append(val.squeeze()) 

            e_state['current_ep_reward'] += reward
            e_state['last_dist'] = 0 if done else d
            total_steps += 1
            
            if done:
                episode_history.append(e_state['current_ep_reward'])
                success_history.append(1.0 if is_success else 0.0)
                distance_history.append(0.0 if is_success else d)
                
                e_state['current_ep_reward'] = 0
                e_state['last_move_time'] = time.time()
                e_state['start_time'] = time.time()
                e_state['off_path_since'] = None
                e_state['flipped_since'] = None
                
                if len(episode_history) % 10 == 0:
                    plt.figure(figsize=(15, 10))
                    
                    plt.subplot(2, 3, 1)
                    plt.plot(episode_history[-100:])
                    plt.title("Episode Reward (last 100)")
                    
                    plt.subplot(2, 3, 2)
                    recent_succ = success_history[-100:]
                    plt.plot([np.mean(recent_succ[:i+1]) for i in range(len(recent_succ))])
                    plt.title(f"Success Rate (last 100): {np.mean(recent_succ):.2f}")
                    
                    plt.subplot(2, 3, 3)
                    plt.plot(distance_history[-100:])
                    plt.title("Failure Distances")
                    
                    if kl_history:
                        plt.subplot(2, 3, 4)
                        plt.plot(kl_history[-100:])
                        plt.title("KL Divergence")
                        
                    if entropy_history:
                        plt.subplot(2, 3, 5)
                        plt.plot(entropy_history[-100:])
                        plt.title("Entropy")
                        
                    if critic_loss_history:
                        plt.subplot(2, 3, 6)
                        plt.plot(critic_loss_history[-100:])
                        plt.title("Critic Loss")
                    
                    plt.tight_layout()
                    plt.savefig('plots/progress.png')
                    plt.close()
                    
                    recent_sr = np.mean(recent_succ)
                    if len(success_history) >= 100 and recent_sr > best_success_rate:
                        best_success_rate = recent_sr
                        print(f"🏆 New Best Success Rate! {best_success_rate:.2f}")
                        save_checkpoint(total_steps)

            if len(states) >= UPDATE_INTERVAL: train()
            
        return jsonify({"action": action.item(), "reset": done})
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return jsonify({"action": 3, "reset": True})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)