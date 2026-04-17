import mujoco
import numpy as np
import time
import os
import urllib.request

# Define paths
model_dir = os.path.expanduser("~/.mujoco/models")
xml_path = os.path.join(model_dir, "humanoid.xml")

# Create directory if needed
os.makedirs(model_dir, exist_ok=True)

# Download humanoid model if not exists
if not os.path.exists(xml_path):
    print("Downloading humanoid model...")
    url = "https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml"
    try:
        urllib.request.urlretrieve(url, xml_path)
        print(f"Downloaded to {xml_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        # Try alternative URL
        url = "https://github.com/google-deepmind/mujoco/raw/refs/heads/main/mjx/mujoco/mjx/test_data/humanoid/humanoid.xml"
        urllib.request.urlretrieve(url, xml_path)
        print(f"Downloaded from alternative URL")

print(f"Using XML at: {xml_path}")

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset to standing pose
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

print("\nHumanoid loaded successfully!")
print(f"Number of joints: {model.njnt}")
print(f"Number of actuators: {model.nu}")
print(f"Simulation timestep: {model.opt.timestep}")

# Simple controller
def walk_controller(t):
    ctrl = np.zeros(model.nu)
    freq = 2.0
    amp = 0.3
    
    if model.nu >= 21:
        ctrl[3] = amp * np.sin(freq * t)
        ctrl[5] = 0.2 * np.sin(freq * t - 1)
        ctrl[7] = amp * np.sin(freq * t + np.pi)
        ctrl[9] = 0.2 * np.sin(freq * t + np.pi - 1)
    
    return ctrl

print("\nRunning 5 second simulation...")
start = time.time()

while data.time < 5.0:
    data.ctrl[:] = walk_controller(data.time)
    mujoco.mj_step(model, data)
    
    seconds = int(data.time)
    if seconds > int(data.time - model.opt.timestep):
        print(f"Time: {data.time:.1f}s, Height: {data.qpos[2]:.2f}m")

print(f"\nDone! Simulated {data.time:.1f}s in {time.time()-start:.1f} real seconds")
