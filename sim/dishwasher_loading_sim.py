# sim/dishwasher_loading_sim.py
import os
import sys
import math
import random
import csv
import time
from collections import deque
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Configuration parameters for dishwasher basket workspace
EPISODE_LENGTH = 160                 # Time budget to align and slide plate
GRAVITY = 9.81

# Get absolute paths relative to this file to prevent working directory mismatches
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HISTORY_DIR = os.path.join(PROJECT_ROOT, "history")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "core", "neural_weights")
os.makedirs(os.path.join(HISTORY_DIR, "dishwasher_tests"), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TELEMETRY_FILE = os.path.join(HISTORY_DIR, "dishwasher_tests", "dishwasher_telemetry.csv")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "dishwasher_ppo.pt")

TELEMETRY_FILE = os.path.join(HISTORY_DIR, "dishwasher_tests", "dishwasher_telemetry.csv")

class DishwasherLoadingEnvironment:
    """
    Simulates a robot arm holding a fragile porcelain plate and attempting to
    insert it blind between springy, narrow metal tines in a dishwasher rack.
    Enforces active variable joint impedance (stiffness Kp) limits: if Kp is too
    high, misalignment shatters the plate. If Kp is soft, the tines safely guide
    the plate into position.
    """
    def __init__(self):
        # Physical metrics for standard plate
        self.plate_mass = 0.50               # 500 grams (porcelain plate weight)
        self.plate_radius = 0.12             # 12cm radius (meters)
        self.plate_crush_limit = 45.0        # Force past this shatters ceramic (Newtons)
        
        # Dishwasher rack slot geometry (meters)
        self.slot_center_x = 0.0             # Target insertion center
        self.slot_width = 0.04               # Width between tines (4cm)
        self.tine_height = 0.10              # Height of metal tines (10cm)
        
        # Springy metal tine physics
        self.k_tines = 850.0                 # Lateral stiffness of metal tines (N/m)
        self.c_tines = 12.0                  # Tine damping
        
        # Base arm joint compliance (impedance variables)
        self.k_arm_default = 180.0           # Default stiff holding position gain (N/m)
        self.min_arm_stiffness = 25.0        # Softest compliant bound (prevent limp drop)
        self.max_arm_stiffness = 400.0       # Max rigid hold
        
        # Physical Guard Rail Compression Stops
        self.max_tine_deflection = 0.005     # 5mm max tine deflection (solid limit stop)
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.plate_shattered = False
        self.is_completed = False
        
        # Gripper coordinate positions [X, Y, Z]
        # Starts 12cm high above rack with a random horizontal misalignment
        start_misalignment = random.uniform(-0.035, 0.035)
        self.plate_pos = np.array([start_misalignment, 0.0, 0.12])
        self.plate_vel = np.zeros(3)
        
        # Gripper holding clamp force
        self.grip_force = 12.0               # Newtons
        
        # Sensed physical forces
        self.tine_lateral_force_x = 0.0
        self.tine_normal_force_z = 0.0
        self.active_arm_stiffness = self.k_arm_default
        
        return self._get_observation()

    def _get_observation(self):
        # Local relative distances to target slot insert center
        obs = [
            self.plate_pos[0] - self.slot_center_x,  # X-axis misalignment
            self.plate_pos[1],                       # Y error
            self.plate_pos[2],                       # Z clearance altitude
            self.plate_vel[0] / 1.5,                 # Velocities
            self.plate_vel[1] / 1.5,
            self.plate_vel[2] / 1.5,
            self.tine_lateral_force_x / self.plate_crush_limit, # Sensed tine forces
            self.tine_normal_force_z / self.plate_crush_limit,
            self.grip_force / self.plate_crush_limit,
            self.active_arm_stiffness / self.max_arm_stiffness, # Current compliance Multiplier
            float(self.plate_shattered),
            float(self.is_completed)
        ]
        
        # Add emulated 4x4 spatial tactile matrix pressure map (12 + 12 = 24D State)
        taxel_pressures = np.zeros(12)
        if abs(self.plate_pos[0]) > (self.slot_width / 2.0) and self.plate_pos[2] < self.tine_height:
            taxel_pressures.fill(self.tine_lateral_force_x / 12)
            
        obs.extend(list(taxel_pressures))
        return np.array(obs, dtype=np.float32)

    def step(self, action, dt=0.032):
        self.current_step += 1
        
        # Save previous Z coordinates to award delta vertical progress
        prev_z = self.plate_pos[2]
        
        # Parse actions
        cmd_vel = np.clip(action[0:3], -1.0, 1.0) * 0.15 # Max command speed 15cm/s
        self.grip_force = max(1.0, min(60.0, (action[3] + 1.0) * 30.0))
        
        # Clamped action scale multiplier
        action_kp_clamped = np.clip(action[4], -1.0, 1.0)
        cmd_kp_scale = (action_kp_clamped + 1.0) * 0.50 # [0.0, 1.0] scale multiplier
        self.active_arm_stiffness = self.min_arm_stiffness + cmd_kp_scale * (self.max_arm_stiffness - self.min_arm_stiffness)
        
        # 14-substep numerical integration (sub_dt = 2.2 ms) to handle stiff spring impacts
        num_substeps = 14
        sub_dt = dt / num_substeps
        
        for _ in range(num_substeps):
            if self.plate_shattered:
                break
                
            # Update coordinate positions using explicit integration
            # Compliant displacement term: dx = F_external / K_arm
            compliant_dx = -self.tine_lateral_force_x / self.active_arm_stiffness
            compliant_dz = -self.tine_normal_force_z / self.active_arm_stiffness
            
            self.plate_pos[0] += (cmd_vel[0] + compliant_dx) * sub_dt
            self.plate_pos[1] += cmd_vel[1] * sub_dt
            self.plate_pos[2] = max(-0.02, min(0.15, self.plate_pos[2] + (cmd_vel[2] + compliant_dz) * sub_dt))
            
            # --- EVALUATE TINE IMPACTS & CONTACT FORCES ---
            if self.plate_pos[2] < self.tine_height:
                if self.plate_pos[2] < 0.0:
                    penetration_z = -self.plate_pos[2]
                    self.tine_normal_force_z = 1500.0 * penetration_z - 5.0 * self.plate_vel[2]
                    self.tine_normal_force_z = max(0.0, self.tine_normal_force_z)
                else:
                    self.tine_normal_force_z = 0.0
                    
                left_boundary = self.slot_center_x - (self.slot_width / 2.0)
                right_boundary = self.slot_center_x + (self.slot_width / 2.0)
                
                # Guard Rail hard position locks to prevent infinite penetration
                if self.plate_pos[0] < left_boundary:
                    deflection_x = left_boundary - self.plate_pos[0]
                    if deflection_x > self.max_tine_deflection:
                        self.plate_pos[0] = left_boundary - self.max_tine_deflection
                        self.plate_vel[0] = 0.0
                        deflection_x = self.max_tine_deflection
                    self.tine_lateral_force_x = self.k_tines * deflection_x - self.c_tines * self.plate_vel[0]
                elif self.plate_pos[0] > right_boundary:
                    deflection_x = self.plate_pos[0] - right_boundary
                    if deflection_x > self.max_tine_deflection:
                        self.plate_pos[0] = right_boundary + self.max_tine_deflection
                        self.plate_vel[0] = 0.0
                        deflection_x = self.max_tine_deflection
                    self.tine_lateral_force_x = -self.k_tines * deflection_x - self.c_tines * self.plate_vel[0]
                else:
                    self.tine_lateral_force_x = 0.0
            else:
                self.tine_normal_force_z = 0.0
                self.tine_lateral_force_x = 0.0
                
            # Evaluate physical damage limits
            net_impact_force = math.sqrt(self.tine_lateral_force_x**2 + self.tine_normal_force_z**2)
            if net_impact_force > self.plate_crush_limit:
                self.plate_shattered = True
                break

        # --- REWARD FUNCTION FORMULATION (CONDITIONED ON ALIGNMENT) ---
        reward = 0.0
        
        # 1. Horizontal Alignment Factor: scales exponential decay as X error increases
        alignment_factor = math.exp(-600.0 * (self.plate_pos[0] ** 2))
        
        # 2. Downward Progress Reward: ONLY awarded if horizontal X alignment is secure!
        z_progress_delta = max(0.0, prev_z - self.plate_pos[2])
        progress_reward = z_progress_delta * 2500.0 * alignment_factor
        
        # 3. Continuous Global Alignment Penalty (enforced at ALL Z altitudes)
        alignment_penalty = -800.0 * (self.plate_pos[0] ** 2)
            
        # 4. Compliance Reward: rewards softening Kp when near or between tines
        compliance_reward = 0.0
        if self.plate_pos[2] < self.tine_height:
            stiffness_ratio = (self.active_arm_stiffness - self.min_arm_stiffness) / (self.max_arm_stiffness - self.min_arm_stiffness)
            compliance_reward = 20.0 * (1.0 - stiffness_ratio)
            
        # 5. Continuous Force Penalty: directly penalizes high contact forces to eliminate 30N impacts
        contact_force_sq = (self.tine_lateral_force_x ** 2) + (self.tine_normal_force_z ** 2)
        force_penalty = -0.15 * contact_force_sq

        reward = progress_reward + alignment_penalty + compliance_reward + force_penalty - 0.05 * np.sum(action ** 2)
        
        if self.plate_shattered:
            reward -= 500.0
        elif self.plate_pos[2] <= 0.005 and abs(self.plate_pos[0]) < (self.slot_width / 2.0):
            self.is_completed = True
            reward += 1200.0
            
        done = (self.current_step >= EPISODE_LENGTH) or self.plate_shattered or self.is_completed
        
        next_obs = self._get_observation()
        return next_obs, reward, done, self.is_completed

class ActorCritic(nn.Module):
    def __init__(self, state_dim=24, action_dim=5, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.tanh1 = nn.Tanh()
        self.gru = nn.GRUCell(128, hidden_dim)
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state, hx):
        features = self.tanh1(self.fc1(state))
        hn = self.gru(features, hx)
        means = self.actor_mean(hn)
        std = torch.exp(self.log_std)
        value = self.critic(hn)
        return means, std, value, hn

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.hxs = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.hxs.clear()

class PPOAgent:
    def __init__(self, state_dim=24, action_dim=5, hidden_dim=64, lr=1.5e-4, gamma=0.99, K_epochs=3, eps_clip=0.2):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.hidden_dim = hidden_dim
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = RolloutBuffer()

    def select_action(self, state, hx):
        with torch.no_grad():
            means, stds, value, next_hx = self.policy_old(state, hx)
            dist = Normal(means, stds)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1), next_hx

    def update(self):
        # Convert rollout buffers to PyTorch Float32 tensors safely
        old_states = torch.stack(self.buffer.states).float()       
        old_actions = torch.stack(self.buffer.actions).float()     
        old_log_probs = torch.stack(self.buffer.log_probs).float() 
        old_values = torch.stack(self.buffer.values).float()       
        old_hxs = torch.stack(self.buffer.hxs).float()             
        
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        
        discounted_returns = []
        discounted_reward = torch.zeros(1)
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = torch.zeros(1)
            discounted_reward = r + (self.gamma * discounted_reward)
            discounted_returns.insert(0, discounted_reward.clone())
            
        returns = torch.stack(discounted_returns).squeeze(-1).float() 
        advantages = returns - old_values.squeeze(-1)         
        
        if len(returns) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        T = old_states.shape[0]
        
        for _ in range(self.K_epochs):
            hx = old_hxs[0].clone()
            
            log_probs_list = []
            entropy_list = []
            values_list = []
            
            for t in range(T):
                if t > 0:
                    mask = (1.0 - float(dones[t-1]))
                    hx = hx * mask
                    
                means, stds, val, hx = self.policy(old_states[t], hx)
                dist = Normal(means, stds)
                
                step_log_prob = dist.log_prob(old_actions[t]).sum(dim=-1)
                step_entropy = dist.entropy().sum(dim=-1)
                
                log_probs_list.append(step_log_prob)
                entropy_list.append(step_entropy)
                values_list.append(val.squeeze(-1))
                
            log_probs = torch.stack(log_probs_list).squeeze(-1).float()
            dist_entropy = torch.stack(entropy_list).squeeze(-1).float()
            state_values = torch.stack(values_list).squeeze(-1).float()
            
            ratios = torch.exp(log_probs - old_log_probs.squeeze(-1))
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * nn.MSELoss()(state_values.view(-1), returns.view(-1))
            entropy_loss = -0.01 * dist_entropy
            
            loss = actor_loss + critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

async def run_dishwasher_pipeline():
    print("==================================================================")
    print("      CHRISTENSON ROBOTICS - BIMANUAL DISHWASHER LOADING GYM      ")
    print("==================================================================")
    print("[Pipeline] Mode: INFINITE TRAINING (Press Ctrl + C to stop)")
    print("[Pipeline] Initializing bimanual GRU Actor-Critic network...")
    print("[Pipeline] Dynamic tine spring-deflection mechanics: Loaded.")
    print("[Pipeline] Active Variable Impedance compliance matching: ONLINE.")
    
    env = DishwasherLoadingEnvironment()
    agent = PPOAgent()

    # Load existing trained weights if present on disk
    if os.path.exists(CHECKPOINT_FILE):
        try:
            agent.policy.load_state_dict(torch.load(CHECKPOINT_FILE, weights_only=True))
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            print(f"[Checkpoint] Loaded expert weights from: {CHECKPOINT_FILE}")
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load weights ({e}). Starting fresh.")
    else:
        print(f"[Checkpoint] No existing checkpoint found at {CHECKPOINT_FILE}. Starting fresh.")
    
    with open(TELEMETRY_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "final_completion_status", "peak_tine_force_n", "final_stiffness_kp", "reward"])
        
    print(f"[Telemetry Logging] File initialized: {TELEMETRY_FILE}\n")
    
    ep = 0
    rolling_reward = -200.0
    recent_successes = deque(maxlen=50)
    update_batch_episodes = 4  # Batch updates across 4 episodes to stabilize PPO gradients
    
    try:
        while True:
            ep += 1
            state = env.reset()
            hx_state = torch.zeros(1, 64) 
            
            episode_reward = 0.0
            done = False
            max_force = 0.0
            ep_start_time = time.time()
            step_count = 0
            
            while not done:
                step_count += 1
                state_t = torch.FloatTensor(state).unsqueeze(0)
                
                action, log_prob, val, next_hx = agent.select_action(state_t, hx_state)
                action_np = action.cpu().numpy().squeeze(0)
                next_state, reward, done, completed_status = env.step(action_np, dt=0.032)
                
                max_force = max(max_force, abs(env.tine_lateral_force_x), abs(env.tine_normal_force_z))
                
                agent.buffer.states.append(state_t.squeeze(0))
                agent.buffer.actions.append(action.detach().squeeze(0))
                agent.buffer.log_probs.append(log_prob.detach())
                agent.buffer.rewards.append(torch.tensor([float(reward)], dtype=torch.float32))
                agent.buffer.dones.append(torch.tensor([done], dtype=torch.bool))
                agent.buffer.values.append(val.detach())
                agent.buffer.hxs.append(hx_state.detach().squeeze(0))
                
                hx_state = next_hx.detach()
                state = next_state
                episode_reward += reward
                
            ep_duration = time.time() - ep_start_time
            sps = step_count / max(0.001, ep_duration)
            
            rolling_reward = 0.90 * rolling_reward + 0.10 * episode_reward
            completed_flag = int(env.is_completed)
            recent_successes.append(completed_flag)
            success_rate = (sum(recent_successes) / len(recent_successes)) * 100.0
            
            status_str = "CONVERGED 🏆" if success_rate >= 95.0 and len(recent_successes) >= 30 else "TRAINING..."
            
            print(f"Episode {ep:04d} | Speed: {sps:6.1f} SPS | Success Rate (50-ep): {success_rate:5.1f}% [{status_str}]")
            print(f"  - Insertion Status: {bool(completed_flag)} | Peak Force: {max_force:.2f}N | Arm Stiffness: {env.active_arm_stiffness:.1f} N/m")
            print(f"  - Episode Reward:   {episode_reward:7.2f} (Rolling Mean: {rolling_reward:7.2f})")
            print("  ------------------------------------------------------------------")
            
            with open(TELEMETRY_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ep, completed_flag, f"{max_force:.3f}", f"{env.active_arm_stiffness:.2f}", f"{episode_reward:.2f}"])
                
            # Perform PPO backpropagation update every 4 episodes to prevent policy collapse
            if ep % update_batch_episodes == 0:
                agent.update()
                torch.save(agent.policy.state_dict(), CHECKPOINT_FILE)
            
    except KeyboardInterrupt:
        # Save model weights immediately on exit
        torch.save(agent.policy.state_dict(), CHECKPOINT_FILE)
        print(f"\n[Checkpoint] Saved trained policy weights to {CHECKPOINT_FILE}")
        print("[Pipeline] Training paused cleanly by user (Ctrl + C). Telemetry files closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_dishwasher_pipeline())
    except KeyboardInterrupt:
        pass
