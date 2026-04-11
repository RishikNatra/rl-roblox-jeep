import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# --- CONFIG (Must match Training Config) ---
STATE_DIM = 9   
ACTION_DIM = 4  
MODEL_PATH = "jeep_model.pth"

class PPOBrain(nn.Module):
    def __init__(self):
        super(PPOBrain, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor = nn.Linear(128, ACTION_DIM)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

# Load the brain
device = torch.device("cpu")
model = PPOBrain().to(device)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode
    print(f"✅ SUCCESSFULLY LOADED BRAIN: {MODEL_PATH}")
    print("🚗 The Jeep is now in INFERENCE MODE (No learning, just driving).")
else:
    print("❌ ERROR: No jeep_model.pth found. You must train the model first!")

@app.route('/act', methods=['POST'])
def act():
    try:
        data = request.json
        # Extract data from Roblox
        d = data['distance']
        a = data['angle']
        s = data['speed']
        sensors = data['sensors']
        
        # Ground sensor (index 5) detection for the river/cliffs
        is_falling = sensors[5] < 3.0
        is_collision = data.get('collision', False)

        # Normalize State exactly as we did in training
        s_norm = [v/60.0 for v in sensors]
        state_t = torch.FloatTensor([*s_norm, d/200.0, a, s/100.0]).to(device)
        
        with torch.no_grad():
            probs, _ = model(state_t)
            # CRITICAL CHANGE: Always pick the action with the HIGHEST probability
            # No more random sampling!
            action = torch.argmax(probs).item()

        # Send back the action. We reset if it crashes or falls, 
        # but since we aren't training, we don't save any data.
        reset_required = (d < 15 or is_collision or is_falling)
        
        return jsonify({
            "action": action, 
            "reset": reset_required
        })
    except Exception as e:
        print(f"⚠️ Inference Error: {e}")
        return jsonify({"action": 3, "reset": True})

if __name__ == '__main__':
    # No training happens here, so we can run this at high speed
    app.run(host='127.0.0.1', port=5000, threaded=False)