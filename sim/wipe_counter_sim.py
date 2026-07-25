# ==============================================================================
# CHRISTENSON ROBOTICS - HYBRID FORCE-POSITION WIPING GYM (v2.9 Speed-Instrumented)
# ==============================================================================
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

TABLE_LENGTH = 1.2       # Length of the countertop in meters
EPISODE_LENGTH = 150     # Max simulation steps per episode
GRAVITY = 9.81

# Resolve workspace directory paths for telemetry and model checkpoints
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HISTORY_DIR = os.path.join(PROJECT_ROOT, "history")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "core", "neural_weights")
os.makedirs(os.path.join(HISTORY_DIR, "wipe_tests"), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TELEMETRY_FILE = os.path.join(HISTORY_DIR, "wipe_tests", "wipe_telemetry_history.csv")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "wipe_counter_ppo.pt")


class CountertopEnvironment:
    """
    Simulates a non-planar countertop surface with dynamic sponge hydration,
    grime saturation, and hybrid force-position subspace projections.
    
    The agent must maintain an optimal contact force window [5.0N to 8.0N]
    while wiping grime off an uneven surface.
    """
    def __init__(self):
        self.num_grid_points = 120
        self.dx = TABLE_LENGTH / self.num_grid_points
        self.table_heights = np.zeros(self.num_grid_points)
        self.grime_densities = np.zeros(self.num_grid_points)
        
        # Construct sinusoidal countertop waves (represents an uneven table surface)
        for i in range(self.num_grid_points):
            x = i * self.dx
            self.table_heights[i] = 0.03 * math.sin(2.0 * math.pi * x / TABLE_LENGTH)
            
        self.reset()

    def reset(self):
        self.sponge_x = 0.05               # Starts near left end of table (m)
        self.sponge_z = 0.10               # Lifted slightly above the surface (m)
        self.sponge_velocity_x = 0.0       # Horizontal sliding speed
        self.sponge_velocity_z = 0.0       # Vertical speed
        self.normal_force = 0.0            # Dynamic vertical contact force (N)
        
        # Re-seed fresh grime across the entire countertop on every single reset
        for i in range(self.num_grid_points):
            self.grime_densities[i] = random.uniform(0.5, 1.0)
            
        self.sponge_water = 1.0            # Hydration level (Starts fully wet)
        self.sponge_grime = 0.0            # Absorbed grime saturation
        self.cleaned_ratio = 0.0
        self.current_step = 0
        self.is_completed = False
        
        self.target_vel = 0.0
        self.target_force = 6.5
        self.kp0 = 150.0
        self.kp1 = 120.0
        
        self.mu_dry = 0.85
        self.mu_fluid = 0.12
        self.alpha = 2.5
        
        return self._get_observation()

    def _get_observation(self):
        grid_idx = max(0, min(self.num_grid_points - 1, int(self.sponge_x / self.dx)))
        local_height = self.table_heights[grid_idx]
        
        obs = [
            self.normal_force / 15.0,             # Realized tactile normal force
            self.target_force / 15.0,             # Policy force target
            self.sponge_velocity_x / 1.5,         # Sliding speed
            self.target_vel / 1.5,                # Policy velocity target
            local_height * 10.0,                  # Local terrain height change
            self.sponge_z * 10.0,                 # Sponge vertical elevation
            self.sponge_x / TABLE_LENGTH,         # Sponge horizontal progress ratio
            self.sponge_water,                    # Sponge hydration level
            self.sponge_grime,                    # Sponge grime load
            self.kp0 / 300.0,                     # Joint 0 stiffness scale
            self.kp1 / 240.0,                     # Joint 1 stiffness scale
            float(self.current_step) / EPISODE_LENGTH # Time ratio
        ]
        return np.array(obs, dtype=np.float32)

    def step(self, action, dt=0.032):
        self.current_step += 1
        prev_cleaned_ratio = self.cleaned_ratio
        
        # Calibrated speed range [0.05 to 0.45 m/s] for controlled, thorough wiping
        self.target_vel = max(0.05, min(0.45, 0.25 + action[0] * 0.20))
        self.target_force = max(2.0, min(12.0, 6.5 + action[1] * 3.5))
        z_regulator = action[2]
        
        self.kp0 = 150.0 + (action[0] * 35.0)
        self.kp1 = 120.0 + (action[1] * 30.0)
        
        num_substeps = 12
        sub_dt = dt / num_substeps
        
        for _ in range(num_substeps):
            grid_idx = max(0, min(self.num_grid_points - 1, int(self.sponge_x / self.dx)))
            surface_height = self.table_heights[grid_idx]
            
            # 1. Position Subspace (X): Slide horizontally
            accel_x = (self.target_vel - self.sponge_velocity_x) / 0.10
            self.sponge_velocity_x += accel_x * sub_dt
            self.sponge_x = max(0.01, min(TABLE_LENGTH - 0.01, self.sponge_x + self.sponge_velocity_x * sub_dt))
            
            # 2. Force Subspace (Z): Closed-loop surface terrain tracking
            k_sponge = 450.0  # N/m
            # Calculate target Z penetration required to achieve target normal force
            ideal_intrusion = self.target_force / k_sponge
            target_z = surface_height - ideal_intrusion + (z_regulator * 0.002)
            
            # Smoothly damp vertical elevation towards target height
            self.sponge_z += 0.20 * (target_z - self.sponge_z)
            
            sponge_intrusion = max(0.0, surface_height - self.sponge_z)
            if self.sponge_z <= surface_height:
                self.normal_force = k_sponge * sponge_intrusion
            else:
                self.normal_force = 0.0
            
            # Water wringing decay: 6.5N pressing force gradually wrings water across sweep
            decay_rate = 0.08 * (self.normal_force / 6.5)
            self.sponge_water = max(0.10, self.sponge_water - decay_rate * sub_dt)
            
            # Boosted grime scrubbing power (18.0 factor) to enable 90%+ cleaning on a single controlled pass
            if self.normal_force >= 1.5:
                clean_power = 18.0 * (self.normal_force / 6.5) * sub_dt
                grime_removed = min(self.grime_densities[grid_idx], clean_power)
                
                self.grime_densities[grid_idx] -= grime_removed
                self.sponge_grime = min(1.0, self.sponge_grime + grime_removed * 0.25)
                
            self.cleaned_ratio = 1.0 - (np.sum(self.grime_densities) / self.num_grid_points)
        
        # Stribeck wet friction calculation
        v_slip = abs(self.sponge_velocity_x)
        viscosity_eff = 0.05 * self.sponge_water
        stribeck = (viscosity_eff * v_slip) / max(0.1, self.normal_force)
        mu = self.mu_fluid + (self.mu_dry - self.mu_fluid) * math.exp(-self.alpha * stribeck)
        friction_force = mu * self.normal_force * np.sign(self.sponge_velocity_x)
        
        delta_cleaned = self.cleaned_ratio - prev_cleaned_ratio
        progress_reward = delta_cleaned * 5000.0
        
        force_penalty = 0.0
        if self.normal_force < 5.0:
            force_penalty = abs(5.0 - self.normal_force) * 2.0
        elif self.normal_force > 8.0:
            force_penalty = abs(self.normal_force - 8.0) * 4.0
            
        reward = progress_reward - (force_penalty * 0.25) - (abs(friction_force) * 0.01) - 0.02 * np.sum(action ** 2)
        
        # Completion check: cleaned 85%+ of grime and reached end of table
        if self.cleaned_ratio >= 0.85 and self.sponge_x >= (TABLE_LENGTH - 0.05):
            self.is_completed = True
            reward += 1000.0
            
        done = (self.current_step >= EPISODE_LENGTH) or (self.sponge_x >= (TABLE_LENGTH - 0.02)) or self.is_completed
        
        next_obs = self._get_observation()
        return next_obs, reward, done, self.is_completed


class ActorCritic(nn.Module):
    def __init__(self, state_dim=12, action_dim=3, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.tanh1 = nn.Tanh()
        self.gru = nn.GRUCell(128, hidden_dim)
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Initialized log_std to -1.2 (std ~0.30) to prevent random sampling noise from overwhelming learned action means
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.2)
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
    def __init__(self, state_dim=12, action_dim=3, hidden_dim=64, lr=3e-4, gamma=0.99, K_epochs=3, eps_clip=0.2):
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


async def run_hybrid_wiping_pipeline():
    print("==================================================================")
    print("      CHRISTENSON ROBOTICS - COUNTERTOP WIPING GYM (INFINITE)    ")
    print("==================================================================")
    print("[Pipeline] Mode: INFINITE TRAINING (Press Ctrl + C to stop)")
    print("[Pipeline] Initializing GRU Neural Actor-Critic controllers...")
    print("[Pipeline] Dynamic normal force target window: 5.0N to 8.0N")
    
    env = CountertopEnvironment()
    agent = PPOAgent()
    
    # Load existing trained weights if present
    if os.path.exists(CHECKPOINT_FILE):
        try:
            agent.policy.load_state_dict(torch.load(CHECKPOINT_FILE, weights_only=True))
            # Clamp log_std to max -1.0 (std <= 0.36) on load so expert weights execute tightly without sampling jitter
            with torch.no_grad():
                agent.policy.log_std.clamp_(max=-1.0)
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            print(f"[Checkpoint] Loaded expert weights from: {CHECKPOINT_FILE}")
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load weights ({e}). Starting fresh.")
    else:
        print(f"[Checkpoint] No existing checkpoint found at {CHECKPOINT_FILE}. Starting fresh.")
    
    with open(TELEMETRY_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "final_completion_status", "cleaned_percentage", "normal_force_n", "reward", "steps_per_sec"])
        
    print(f"[Telemetry Logging] Telemetry initialized at: {TELEMETRY_FILE}\n")
    
    ep = 0
    rolling_reward = -350.0
    recent_successes = deque(maxlen=50)
    update_batch_episodes = 4
    
    try:
        while True:
            ep += 1
            state = env.reset()
            hx_state = torch.zeros(1, 64) 
            
            episode_reward = 0.0
            done = False
            ep_start_time = time.time()
            step_count = 0
            
            while not done:
                step_count += 1
                state_t = torch.FloatTensor(state).unsqueeze(0)
                
                action, log_prob, val, next_hx = agent.select_action(state_t, hx_state)
                action_np = action.cpu().numpy().squeeze(0)
                next_state, reward, done, completed_status = env.step(action_np, dt=0.032)
                
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
            print(f"  - Cleaned Grime: {env.cleaned_ratio * 100.0:5.1f}% | Normal Force: {env.normal_force:.2f}N | Sponge Water: {env.sponge_water * 100.0:.1f}%")
            print(f"  - Episode Reward:   {episode_reward:7.2f} (Rolling Mean: {rolling_reward:7.2f})")
            print("  ------------------------------------------------------------------")
            
            with open(TELEMETRY_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ep, completed_flag, f"{env.cleaned_ratio*100.0:.1f}", f"{env.normal_force:.2f}", f"{episode_reward:.2f}", f"{sps:.1f}"])
                
            if ep % update_batch_episodes == 0:
                agent.update()
                torch.save(agent.policy.state_dict(), CHECKPOINT_FILE)
            
    except KeyboardInterrupt:
        torch.save(agent.policy.state_dict(), CHECKPOINT_FILE)
        print(f"\n[Checkpoint] Saved trained policy weights to {CHECKPOINT_FILE}")
        print("[Pipeline] Wiping gym infinite training paused cleanly by user (Ctrl + C). Telemetry files closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_hybrid_wiping_pipeline())
    except KeyboardInterrupt:
        pass
