import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
import os
import math

app = Flask(__name__)

# ============================================================
# WAYPOINT PATHFINDING CONTROLLER
# No RL — pure reactive steering with obstacle avoidance
# Requires updated CarScript that sends waypoint data
# ============================================================

# Steering thresholds
TURN_THRESHOLD = 0.10       # Cross product threshold to start turning
HARD_TURN_THRESHOLD = 0.40  # Cross product threshold for aggressive turning
WAYPOINT_REACH_DIST = 20.0  # Distance to consider waypoint "reached"
OBSTACLE_CLOSE = 8.0        # Sensor distance to trigger obstacle avoidance
OBSTACLE_DANGER = 4.0       # Sensor distance for emergency avoidance

@app.route('/act', methods=['POST'])
def act():
    try:
        data = request.json
        
        sensors = data.get('sensors', [60]*6)
        while len(sensors) < 6: sensors.append(60.0)
        
        speed = data.get('speed', 0.0)
        is_collision = data.get('collision', False)
        is_flipped = data.get('flipped', False)
        
        # Waypoint data (sent by updated CarScript)
        wp_angle = data.get('waypointAngle', 0.0)    # Dot product: 1.0 = facing wp, -1.0 = facing away
        wp_cross = data.get('waypointCross', 0.0)     # Cross.Y: >0 = wp is LEFT, <0 = wp is RIGHT
        wp_dist  = data.get('waypointDistance', 999.0) # Distance to current waypoint
        wp_index = data.get('currentWaypoint', 1)      # Which waypoint we're heading to
        wp_total = data.get('totalWaypoints', 14)      # Total waypoints
        
        # --- Sensor layout ---
        # 0: left (-90°), 1: front-left (-45°), 2: slight-left (-15°)
        # 3: slight-right (15°), 4: front-right (45°), 5: straight ahead (0°)
        s_left, s_fleft, s_sleft, s_sright, s_fright, s_front = sensors
        
        action = compute_steering(
            wp_cross, wp_angle, wp_dist,
            s_left, s_fleft, s_sleft, s_sright, s_fright, s_front,
            speed
        )
        
        # Reset conditions
        reset_required = is_collision or is_flipped
        
        if wp_index > wp_total:
            print(f"🏁 ALL {wp_total} WAYPOINTS REACHED! Resetting.")
            reset_required = True
        
        dirs = ["⬆ Forward", "⬅ Left", "➡ Right", "⏹ Stop"]
        if not reset_required:
            print(f"📍 WP {wp_index}/{wp_total} | Dist: {wp_dist:.1f} | Cross: {wp_cross:+.2f} | Angle: {wp_angle:+.2f} | {dirs[action]} | Sensors: L={s_left:.0f} FL={s_fleft:.0f} F={s_front:.0f} FR={s_fright:.0f} R={s_fright:.0f}")
        else:
            print(f"♻️ RESET (Col={is_collision}, Flip={is_flipped})")
        
        return jsonify({
            "action": action, 
            "reset": reset_required
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Error: {e}")
        return jsonify({"action": 0, "reset": False})


def compute_steering(wp_cross, wp_angle, wp_dist, 
                     s_left, s_fleft, s_sleft, s_sright, s_fright, s_front,
                     speed):
    """
    Reactive steering controller using potential-field-like logic:
    1. First check for obstacles and avoid them
    2. Then steer toward current waypoint using cross product (signed direction)
    """
    
    # ================================================================
    # LAYER 1: EMERGENCY OBSTACLE AVOIDANCE (overrides everything)
    # ================================================================
    
    if s_front < OBSTACLE_DANGER:
        # Something right in front — turn whichever side is more open
        if s_left + s_fleft > s_fright + s_left:
            return 1  # Turn left
        else:
            return 2  # Turn right
    
    # ================================================================
    # LAYER 2: PROACTIVE OBSTACLE AVOIDANCE
    # Blend obstacle avoidance with waypoint steering
    # ================================================================
    
    # Compute obstacle "pressure" from each side
    left_pressure  = max(0, 1.0 - s_fleft / OBSTACLE_CLOSE) + max(0, 1.0 - s_sleft / OBSTACLE_CLOSE) * 0.5
    right_pressure = max(0, 1.0 - s_fright / OBSTACLE_CLOSE) + max(0, 1.0 - s_sright / OBSTACLE_CLOSE) * 0.5
    front_pressure = max(0, 1.0 - s_front / OBSTACLE_CLOSE)
    
    # If significant pressure from one side, dodge
    if front_pressure > 0.3 or left_pressure > 0.5 or right_pressure > 0.5:
        if left_pressure > right_pressure + 0.1:
            return 2  # Obstacle on left → turn right
        elif right_pressure > left_pressure + 0.1:
            return 1  # Obstacle on right → turn left
        elif front_pressure > 0.3:
            # Front blocked, turn toward waypoint side if possible
            if wp_cross > 0:
                return 1  # Waypoint is left, try left
            else:
                return 2
    
    # ================================================================
    # LAYER 3: WAYPOINT STEERING (when path is clear)
    # Uses cross product for direction: + = wp is LEFT, - = wp is RIGHT
    # ================================================================
    
    # If facing almost directly away from waypoint, make a hard turn
    if wp_angle < -0.3:
        # Facing away — need to do a U-turn toward the waypoint
        if wp_cross >= 0:
            return 1  # Turn left
        else:
            return 2  # Turn right
    
    # Normal proportional steering
    if wp_cross > HARD_TURN_THRESHOLD:
        return 1  # Hard left — waypoint is far to the left
    elif wp_cross < -HARD_TURN_THRESHOLD:
        return 2  # Hard right — waypoint is far to the right
    elif wp_cross > TURN_THRESHOLD:
        return 1  # Gentle left
    elif wp_cross < -TURN_THRESHOLD:
        return 2  # Gentle right
    else:
        return 0  # Facing waypoint — go forward


if __name__ == '__main__':
    print("=" * 60)
    print("🗺️  WAYPOINT PATHFINDING MODE")
    print("    Following Map.Path waypoints 1 → 14")
    print("    No RL model needed — pure reactive steering")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, threaded=False)