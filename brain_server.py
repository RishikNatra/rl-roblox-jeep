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
LR_ACTOR = 0.001      # Separate, higher LR for actor to break out of uniform policy
LR_CRITIC = 0.001     # Critic needs to learn fast too
GAMMA = 0.99         
GAE_LAMBDA = 0.95 
PPO_CLIP = 0.2    
ENTROPY_COEFF = 0.01  # Low — we want it to commit to actions, not stay random
UPDATE_INTERVAL = 512  # Bigger buffer = better gradient estimates
MINI_BATCH_SIZE = 64   # Mini-batch SGD within each PPO epoch
PPO_EPOCHS = 10        # More passes over the data with mini-batches
MODEL_PATH = os.path.join(os.path.dirname(__file__), "jeep_model.pth") 


class PPOBrain(nn.Module):
    def __init__(self):
        super(PPOBrain, self).__init__()
        # Separate networks for actor and critic (no shared backbone)
        # This prevents the critic's gradient from interfering with the actor
        self.actor_net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, ACTION_DIM)
        )
        self.critic_net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Orthogonal init — proven to help PPO converge
        for module in [self.actor_net, self.critic_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        # Last actor layer gets small init so initial policy is near-uniform
        nn.init.orthogonal_(self.actor_net[-1].weight, gain=0.01)

    def forward(self, x):
        return torch.softmax(self.actor_net(x), dim=-1), self.critic_net(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor_net(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic_net(x)

device = torch.device("cpu")
model = PPOBrain().to(device)
optimizer = optim.Adam([
    {'params': model.actor_net.parameters(), 'lr': LR_ACTOR},
    {'params': model.critic_net.parameters(), 'lr': LR_CRITIC}
])
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

# Rollout buffer
buf_states, buf_actions, buf_rewards, buf_log_probs, buf_dones, buf_values = [], [], [], [], [], []
episode_history = []
success_history = []
distance_history = []
entropy_history = []
kl_history = []
critic_loss_history = []
actor_loss_history = []
best_success_rate = 0.0

def train():
    global buf_states, buf_actions, buf_rewards, buf_log_probs, buf_dones, buf_values
    if len(buf_states) < UPDATE_INTERVAL: return
    
    S = torch.stack(buf_states).to(device)
    A = torch.tensor(buf_actions, dtype=torch.long).to(device)
    OLD_LP = torch.stack(buf_log_probs).detach().to(device)
    V = torch.stack(buf_values).detach().squeeze(-1).to(device)
    R = torch.tensor(buf_rewards, dtype=torch.float32).to(device)
    D = torch.tensor(buf_dones, dtype=torch.float32).to(device)
    
    # --- Compute GAE advantages ---
    with torch.no_grad():
        N = len(R)
        advantages = torch.zeros(N).to(device)
        last_gae = 0.0
        for t in reversed(range(N)):
            if t == N - 1:
                next_val = 0.0
            else:
                next_val = V[t + 1]
            not_done = 1.0 - D[t]
            delta = R[t] + GAMMA * next_val * not_done - V[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * not_done * last_gae
        returns = advantages + V
        
        # Normalize advantages (NOT returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # --- Mini-batch PPO updates ---
    epoch_kls = []
    epoch_entropies = []
    epoch_critic_losses = []
    epoch_actor_losses = []
    
    indices = np.arange(N)
    
    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        
        for start in range(0, N, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_idx = indices[start:end]
            
            mb_states = S[mb_idx]
            mb_actions = A[mb_idx]
            mb_old_lp = OLD_LP[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            
            # Forward pass
            _, new_lp, entropy, new_val = model.get_action_and_value(mb_states, mb_actions)
            
            # Policy loss
            log_ratio = new_lp - mb_old_lp
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped)
            v_loss = 0.5 * mse_loss(new_val.squeeze(), mb_returns)
            
            # Entropy bonus
            entropy_bonus = entropy.mean()
            
            # Combined loss
            loss = actor_loss + 0.5 * v_loss - ENTROPY_COEFF * entropy_bonus
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Track KL for monitoring (not early stopping)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
            
            epoch_kls.append(approx_kl)
            epoch_entropies.append(entropy_bonus.item())
            epoch_critic_losses.append(v_loss.item())
            epoch_actor_losses.append(actor_loss.item())
        
        # Early stop only if KL is truly catastrophic (not at 0.02!)
        mean_kl = np.mean(epoch_kls[-max(1, N // MINI_BATCH_SIZE):])
        if mean_kl > 0.1:
            print(f"⚠️ KL too high ({mean_kl:.4f}), stopping PPO epochs at epoch {epoch+1}")
            break

    kl_history.append(np.mean(epoch_kls))
    entropy_history.append(np.mean(epoch_entropies))
    critic_loss_history.append(np.mean(epoch_critic_losses))
    actor_loss_history.append(np.mean(epoch_actor_losses))
    
    print(f"📊 Train | KL: {kl_history[-1]:.4f} | Entropy: {entropy_history[-1]:.3f} | CriticL: {critic_loss_history[-1]:.3f} | ActorL: {actor_loss_history[-1]:.3f}")
    
    # Clear buffer
    buf_states, buf_actions, buf_rewards, buf_log_probs, buf_dones, buf_values = [], [], [], [], [], []
    
    # Save checkpoint every 5000+ steps
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
                    'last_dist': None, 
                    'last_g_dist': None,
                    'current_ep_reward': 0, 
                    'steps_in_ep': 0,
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
            e_state['steps_in_ep'] += 1
            
            if s > 2.0: e_state['last_move_time'] = current_time
            is_stuck = (current_time - e_state['last_move_time']) > 10.0

            # Initialize last_dist on first step (no spurious delta)
            if e_state['last_dist'] is None: e_state['last_dist'] = d
            progress = e_state['last_dist'] - d  # Can be negative (moving away)
            
            # Timeout (1 min)
            is_timeout = (current_time - e_state['start_time']) > 60.0
            
            # Off path tracking (> 10s)
            if g_dist > 10.0:
                if e_state['off_path_since'] is None:
                    e_state['off_path_since'] = current_time
            else:
                e_state['off_path_since'] = None
            is_off_path_long = e_state['off_path_since'] is not None and (current_time - e_state['off_path_since']) > 10.0
            
            # Flipped tracking (> 3s — faster reset for flips)
            is_flipped = data.get('flipped', False)
            if is_flipped:
                if e_state['flipped_since'] is None:
                    e_state['flipped_since'] = current_time
            else:
                e_state['flipped_since'] = None
            is_flipped_long = e_state['flipped_since'] is not None and (current_time - e_state['flipped_since']) > 3.0
            
            # ========================================
            # SIMPLE, CLEAR REWARD FUNCTION
            # ========================================
            
            # Primary signal: distance to target decreased = good  
            # This is the ONLY thing that truly matters for convergence
            progress_reward = np.clip(progress * 0.1, -1.0, 1.0)
            
            # Secondary: stay on the guide path  
            # Smooth decay: 1.0 at center, 0.5 at 5 studs, ~0 at 15+
            on_path_reward = np.exp(-g_dist / 5.0)  # Exponential decay, always positive
            
            # Tertiary: face the target
            facing_reward = max(0.0, a) * 0.2  # Only reward facing toward, don't punish facing away (progress_reward handles that)
            
            # Guide delta (getting closer to path)
            if e_state['last_g_dist'] is None: e_state['last_g_dist'] = g_dist
            g_progress = e_state['last_g_dist'] - g_dist
            e_state['last_g_dist'] = g_dist
            returning_to_path_reward = np.clip(g_progress * 2.0, -0.5, 0.5)
            
            # Assemble reward: all components are small and bounded
            reward = progress_reward + on_path_reward + facing_reward + returning_to_path_reward
            
            # Penalties
            if s < 2.0:
                reward -= 0.3  # Don't just sit still
            
            done = False
            is_success = False
            
            if d < 15: 
                reward += 10.0      # Big clear signal for reaching goal
                done = True
                is_success = True
                print(f"✨ [{env_id}] Level {lvl} Success! (steps: {e_state['steps_in_ep']})")
            elif is_off_path_long:
                reward -= 2.0
                done = True
                print(f"🚫 [{env_id}] Off-Path > 10s (g_dist: {g_dist:.1f})")
            elif is_flipped_long:
                reward -= 2.0
                done = True
                print(f"💥 [{env_id}] Flipped > 3s")
            elif is_timeout:
                reward -= 2.0
                done = True
                print(f"⏰ [{env_id}] Timeout (dist: {d:.1f})")
            elif is_collision:
                reward -= 2.0
                done = True
                print(f"💥 [{env_id}] Collision (dist: {d:.1f})")
            elif sensors[5] < 3.0:
                reward -= 2.0
                done = True
                print(f"💥 [{env_id}] Front wall (sensor: {sensors[5]:.1f}, dist: {d:.1f})")
            elif is_stuck:
                reward -= 2.0
                done = True
                print(f"🧱 [{env_id}] Stuck (dist: {d:.1f})")

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
                action, log_prob, _, val = model.get_action_and_value(state_t.unsqueeze(0))
                action = action.squeeze()
                log_prob = log_prob.squeeze()
                val = val.squeeze()

            # Store transition
            buf_states.append(state_t)
            buf_actions.append(action.item())
            buf_rewards.append(reward)
            buf_log_probs.append(log_prob)
            buf_dones.append(1.0 if done else 0.0)
            buf_values.append(val)

            e_state['current_ep_reward'] += reward
            e_state['last_dist'] = d  # Always update (not None on done — we handle init separately)
            total_steps += 1
            
            if done:
                episode_history.append(e_state['current_ep_reward'])
                success_history.append(1.0 if is_success else 0.0)
                distance_history.append(0.0 if is_success else d)
                
                e_state['current_ep_reward'] = 0
                e_state['steps_in_ep'] = 0
                e_state['last_move_time'] = time.time()
                e_state['start_time'] = time.time()
                e_state['off_path_since'] = None
                e_state['flipped_since'] = None
                e_state['last_g_dist'] = None
                e_state['last_dist'] = None
                
                if len(episode_history) % 10 == 0:
                    plt.figure(figsize=(18, 10))
                    
                    plt.subplot(2, 4, 1)
                    plt.plot(episode_history[-200:])
                    plt.title("Episode Reward (last 200)")
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 4, 2)
                    recent_succ = success_history[-200:]
                    window = min(50, len(recent_succ))
                    if len(recent_succ) >= window:
                        rolling = [np.mean(recent_succ[max(0,i-window):i+1]) for i in range(len(recent_succ))]
                        plt.plot(rolling)
                    plt.title(f"Success Rate (50-ep window): {np.mean(success_history[-50:]):.2f}")
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 4, 3)
                    plt.plot(distance_history[-200:])
                    plt.title("Failure Distances")
                    plt.grid(True, alpha=0.3)
                    
                    if kl_history:
                        plt.subplot(2, 4, 4)
                        plt.plot(kl_history[-100:])
                        plt.title("KL Divergence")
                        plt.grid(True, alpha=0.3)
                        
                    if entropy_history:
                        plt.subplot(2, 4, 5)
                        plt.plot(entropy_history[-100:])
                        plt.title("Entropy")
                        plt.grid(True, alpha=0.3)
                        
                    if critic_loss_history:
                        plt.subplot(2, 4, 6)
                        plt.plot(critic_loss_history[-100:])
                        plt.title("Critic Loss")
                        plt.grid(True, alpha=0.3)
                    
                    if actor_loss_history:
                        plt.subplot(2, 4, 7)
                        plt.plot(actor_loss_history[-100:])
                        plt.title("Actor Loss")
                        plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 4, 8)
                    plt.text(0.5, 0.5, 
                             f"Steps: {total_steps}\n"
                             f"Episodes: {len(episode_history)}\n"
                             f"Best SR: {best_success_rate:.2f}\n"
                             f"Entropy Coeff: {ENTROPY_COEFF:.4f}",
                             ha='center', va='center', fontsize=12,
                             transform=plt.gca().transAxes)
                    plt.title("Stats")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig('plots/progress.png')
                    plt.close()
                    
                    recent_sr = np.mean(success_history[-50:])
                    if len(success_history) >= 50 and recent_sr > best_success_rate:
                        best_success_rate = recent_sr
                        print(f"🏆 New Best Success Rate! {best_success_rate:.2f}")
                        save_checkpoint(total_steps)

            if len(buf_states) >= UPDATE_INTERVAL: train()
            
        return jsonify({"action": action.item(), "reset": done})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Error: {e}")
        return jsonify({"action": 3, "reset": True})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)