from flask import Flask, render_template_string, Response, jsonify, request
import mujoco
import numpy as np
import cv2
import threading
import time

app = Flask(__name__)

# Load model
import os
xml_path = os.path.expanduser("~/.mujoco/models/humanoid.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

# Renderer
renderer = mujoco.Renderer(model, height=600, width=800)

# Global state
simulation_running = True
control_mode = "stand"  # stand, walk, push
push_force = [0, 0, 0]
lock = threading.Lock()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MuJoCo Humanoid Simulator</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; margin: 20px; }
        h1 { color: #4CAF50; }
        .container { display: flex; gap: 20px; }
        .video { border: 3px solid #4CAF50; border-radius: 8px; }
        .controls { background: #2a2a2a; padding: 20px; border-radius: 8px; min-width: 300px; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; 
                 margin: 5px; cursor: pointer; border-radius: 4px; font-size: 16px; }
        button:hover { background: #45a049; }
        button.active { background: #ff9800; }
        .stats { background: #333; padding: 10px; margin: 10px 0; border-radius: 4px; }
        .arrow-keys { display: grid; grid-template-columns: repeat(3, 60px); gap: 5px; margin: 10px 0; }
        .arrow-btn { width: 60px; height: 60px; font-size: 24px; }
        .push-info { color: #ff9800; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🤖 MuJoCo Humanoid Simulator</h1>
    <div class="container">
        <div>
            <img src="/video_feed" class="video" width="800" height="600">
        </div>
        <div class="controls">
            <h3>Control Mode</h3>
            <button onclick="setMode('stand')" id="btn-stand" class="active">STAND</button>
            <button onclick="setMode('walk')" id="btn-walk">WALK</button>
            <button onclick="setMode('jump')" id="btn-jump">JUMP</button>
            
            <h3>Push Force (Arrow Keys)</h3>
            <div class="arrow-keys">
                <div></div>
                <button class="arrow-btn" onmousedown="push('up')" onmouseup="push('none')">⬆️</button>
                <div></div>
                <button class="arrow-btn" onmousedown="push('left')" onmouseup="push('none')">⬅️</button>
                <button class="arrow-btn" onmousedown="push('down')" onmouseup="push('none')">⬇️</button>
                <button class="arrow-btn" onmousedown="push('right')" onmouseup="push('none')">➡️</button>
            </div>
            <div class="push-info" id="push-status">No force applied</div>
            
            <h3>Stats</h3>
            <div class="stats">
                <div>Time: <span id="time">0.00</span>s</div>
                <div>Height: <span id="height">0.00</span>m</div>
                <div>Velocity: <span id="velocity">0.00</span>m/s</div>
            </div>
            
            <button onclick="resetSim()" style="background: #f44336;">RESET</button>
        </div>
    </div>

    <script>
        let currentMode = 'stand';
        
        function setMode(mode) {
            currentMode = mode;
            fetch('/set_mode/' + mode);
            document.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-' + mode).classList.add('active');
        }
        
        function push(direction) {
            fetch('/push/' + direction);
            document.getElementById('push-status').innerText = 
                direction === 'none' ? 'No force applied' : 'Pushing: ' + direction;
        }
        
        function resetSim() {
            fetch('/reset');
        }
        
        // Update stats
        setInterval(() => {
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('time').innerText = data.time.toFixed(2);
                document.getElementById('height').innerText = data.height.toFixed(2);
                document.getElementById('velocity').innerText = data.velocity.toFixed(2);
            });
        }, 100);
    </script>
</body>
</html>
"""

def controller():
    """Control logic"""
    global control_mode, push_force
    
    with lock:
        mode = control_mode
        force = push_force.copy()
    
    # Apply push force to torso
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    data.xfrc_applied[torso_id] = force + [0, 0, 0, 0, 0]
    
    ctrl = np.zeros(model.nu)
    
    if mode == "stand":
        # Standing controller
        ctrl = -0.3 * data.qpos[7:] - 0.1 * data.qvel[6:]
        
    elif mode == "walk":
        # Walking controller
        t = data.time
        freq = 2.0
        if model.nu >= 21:
            ctrl[3] = 0.4 * np.sin(freq * t)      # right hip
            ctrl[5] = 0.3 * np.sin(freq * t - 1)  # right knee
            ctrl[7] = 0.4 * np.sin(freq * t + np.pi)  # left hip
            ctrl[9] = 0.3 * np.sin(freq * t + np.pi - 1)  # left knee
            
    elif mode == "jump":
        # Jump controller - bend then extend
        phase = (data.time % 2.0) / 2.0
        if phase < 0.3:  # Crouch
            ctrl[5] = ctrl[9] = 0.5  # knees bent
            ctrl[3] = ctrl[7] = 0.3  # hips
        else:  # Jump up
            ctrl[5] = ctrl[9] = -1.0
            ctrl[3] = ctrl[7] = -0.5
    
    data.ctrl[:] = ctrl

def generate_frames():
    """Video streaming"""
    global simulation_running
    
    while simulation_running:
        # Step simulation
        controller()
        mujoco.mj_step(model, data)
        
        # Render
        renderer.update_scene(data, camera="track")
        img = renderer.render()
        
        # Convert to JPEG
        ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global control_mode
    with lock:
        control_mode = mode
    return jsonify({"status": "ok", "mode": mode})

@app.route('/push/<direction>')
def push(direction):
    global push_force
    force = 100  # Push strength
    with lock:
        if direction == 'up':
            push_force = [force, 0, 0]
        elif direction == 'down':
            push_force = [-force, 0, 0]
        elif direction == 'left':
            push_force = [0, force, 0]
        elif direction == 'right':
            push_force = [0, -force, 0]
        else:
            push_force = [0, 0, 0]
    return jsonify({"status": "ok"})

@app.route('/reset')
def reset():
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    return jsonify({"status": "reset"})

@app.route('/stats')
def stats():
    return jsonify({
        "time": float(data.time),
        "height": float(data.qpos[2]),
        "velocity": float(np.linalg.norm(data.qvel[:3]))
    })

if __name__ == '__main__':
    print("Starting web server...")
    print("Open your browser and go to: http://your-server-ip:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
