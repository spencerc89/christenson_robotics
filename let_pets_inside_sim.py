# ==============================================================================
# CHRISTENSON ROBOTICS - PORTABLE PET-SAFE DOOR ENTRY GYM (v2.9 Speed-Instrumented)
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

# ==============================================================================
# Physical Constants & Workspace Bounds
# ==============================================================================
EPISODE_LENGTH = 320                 # Expanded time budget for full 5-phase completion
GRAVITY = 9.81

# Ensure local directories exist for logging run analytics and checkpoints
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HISTORY_DIR = os.path.join(PROJECT_ROOT, "history")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "core", "neural_weights")
os.makedirs(os.path.join(HISTORY_DIR, "pet_tests"), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TELEMETRY_FILE = os.path.join(HISTORY_DIR, "pet_tests", "pet_entry_telemetry.csv")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "pet_entry_ppo.pt")

class PetInsideEnvironment:
    """
    Simulates a multi-phase pet entry automation task.
    Phase 0: Pressing down the spring-loaded handle to unlatch the door.
    Phase 1: Swinging the door panel wide open (positive angular acceleration).
             If the handle is released before the door is cracked open, the latch re-locks.
    Phase 2: Waiting for the waiting pet to safely cross the threshold.
    Phase 3: Closing the door extremely slowly (Velocity <= 0.04 rad/s).
             IF any resistance torque > 2.0 N*m is felt, the robot halts, immediately
             swings the door open again, waits, and retries.
    Phase 4: Squeezing/Applying high latch torque (up to 8.0 N*m) at the very end
             (Theta <= 0.05 rad) to snap the door latch closed.
    """
    def __init__(self):
        # Door handle physical limits
        self.k_handle = 25.0                 # Handle spring stiffness (N/m)
        self.c_handle = 1.2                  # Handle damping
        self.max_handle_angle = 1.05         # ~60 degrees of rotation max to unlatch
        self.latch_release_angle = 0.85      # Angle required to release the door latch (~50 degrees)
        self.handle_damage_torque = 45.0     # Structural damage limit (N*m)
        
        # Door panel physical dimensions & mechanics
        self.door_width = 0.85               # Door panel width (meters)
        self.door_mass = 12.0                # 12kg wooden door panel
        self.door_inertia = (1.0 / 3.0) * self.door_mass * (self.door_width ** 2) # Rotational inertia (kg * m^2)
        self.hinge_damping = 2.5             # Sticky hinge resistance
        
        # Safety Reflex and Speed limits
        self.safe_closing_speed_limit = 0.04 # Target slow closing speed: 0.04 rad/s (~2.3 deg/s)
        self.resistance_threshold = 2.0      # Torque resistance threshold to trip safety (N*m)
        self.cooldown_steps_target = 35      # Waiting steps to let pet clear before closing again
        self.latch_engage_zone = 0.05        # Trigger high pressure zone within last 4 degrees
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.active_phase = 0                # 0: Unlatch, 1: Opening, 2: Pet Crossing, 3: Slow Close, 4: Latch Snap
        self.is_damaged = False
        self.is_completed = False
        
        # Gripper Hand Positions [X, Y, Z]
        # Hand 0 (Primary) starts resting directly on top of the handle lever
        self.hand_0_pos = np.array([0.72, 0.0, 0.00])
        self.hand_0_vel = np.zeros(3)
        
        # Handle state (Current downward angle, starting level at 0.0)
        self.handle_angle = 0.0
        self.handle_vel = 0.0
        self.applied_handle_torque = 0.0
        
        # Door panel angle (Radians, starting closed at 0.0, open at 1.57)
        self.door_angle = 0.0
        self.door_angular_vel = 0.0
        self.applied_door_torque = 0.0
        self.applied_latch_torque = 0.0
        
        # Pet position along cross-axis (meters, starts outside at -0.6m, target is inside at +0.6m)
        self.pet_pos = -0.60
        self.pet_vel = 0.0
        
        # Safety Reflex and Cooldown Triggers
        self.safety_reflex_tripped = False
        self.cooldown_timer = 0
        self.felt_resistance_torque = 0.0
        self.retry_count = 0
        
        return self._get_observation()

    def step(self, action, dt=0.032):
        self.current_step += 1
        
        reward = 0.0
        prev_door_angle = self.door_angle
        prev_pet_pos = self.pet_pos
        prev_handle_angle = self.handle_angle
        
        target_vel_h0 = np.clip(action[0:3], -1.0, 1.0) * 0.15 
        cmd_handle_torque = max(0.0, min(35.0, (action[3] + 1.0) * 17.5)) 
        cmd_door_torque = np.clip(action[4], -1.0, 1.0) * 15.0  
        
        self.hand_0_vel = 0.85 * self.hand_0_vel + 0.15 * target_vel_h0
        
        num_substeps = 14
        sub_dt = dt / num_substeps
        
        for _ in range(num_substeps):
            if self.is_damaged:
                break
                
            self.hand_0_pos += self.hand_0_vel * sub_dt
            
            if self.hand_0_pos[2] <= 0.0 and abs(self.hand_0_pos[0] - 0.72) < 0.08:
                self.applied_handle_torque = cmd_handle_torque
                
                handle_accel = (self.applied_handle_torque - self.k_handle * self.handle_angle - self.c_handle * self.handle_vel) / 0.15
                self.handle_vel += handle_accel * sub_dt
                self.handle_angle = max(0.0, min(self.max_handle_angle, self.handle_angle + self.handle_vel * sub_dt))
            else:
                self.applied_handle_torque = 0.0
                self.handle_vel = -self.k_handle * self.handle_angle / self.c_handle
                self.handle_angle = max(0.0, self.handle_angle + self.handle_vel * sub_dt)
                
            if self.applied_handle_torque > self.handle_damage_torque:
                self.is_damaged = True
                break
            
            if self.active_phase == 0:
                if self.handle_angle >= self.latch_release_angle:
                    self.active_phase = 1
                    reward += 100.0 # Milestone bonus for unlatching
                    
            elif self.active_phase == 1:
                if self.handle_angle < self.latch_release_angle and self.door_angle < 0.08:
                    self.active_phase = 0
                    self.applied_door_torque = 0.0
                else:
                    self.applied_door_torque = cmd_door_torque
                    
                    damping_torque = -self.hinge_damping * self.door_angular_vel
                    net_door_torque = self.applied_door_torque + damping_torque
                    
                    door_accel = net_door_torque / self.door_inertia
                    self.door_angular_vel += door_accel * sub_dt
                    self.door_angle = max(0.0, min(1.57, self.door_angle + self.door_angular_vel * sub_dt))
                    
                    if self.door_angle >= 1.05:
                        self.active_phase = 2
                        self.cooldown_timer = 0
                        reward += 300.0 # Milestone bonus for opening door wide
                    
            elif self.active_phase == 2:
                self.applied_door_torque = -15.0 * self.door_angular_vel
                
                door_accel = self.applied_door_torque / self.door_inertia
                self.door_angular_vel += door_accel * sub_dt
                self.door_angle = max(1.05, min(1.57, self.door_angle + self.door_angular_vel * sub_dt))
                
                self.pet_vel = 0.45 # Accelerated pet speed (0.45 m/s) to cross in ~2.1 seconds
                self.pet_pos += self.pet_vel * sub_dt
                
                if self.pet_pos >= 0.35:
                    self.active_phase = 3
                    self.safety_reflex_tripped = False
                    reward += 600.0 # Milestone bonus for pet crossing safely
                    
            elif self.active_phase == 3:
                if self.safety_reflex_tripped:
                    self.cooldown_timer += 1
                    # Smoothly swing door back open to let obstacle clear
                    target_opening_torque = 8.0 * (1.20 - self.door_angle) - 2.5 * self.door_angular_vel
                    self.applied_door_torque = target_opening_torque
                    
                    door_accel = self.applied_door_torque / self.door_inertia
                    self.door_angular_vel += door_accel * sub_dt
                    self.door_angle = max(0.0, min(1.57, self.door_angle + self.door_angular_vel * sub_dt))
                    
                    self.pet_pos = max(-0.50, self.pet_pos - 0.15 * sub_dt)
                    
                    if self.cooldown_timer >= self.cooldown_steps_target:
                        self.safety_reflex_tripped = False
                        self.cooldown_timer = 0
                        self.felt_resistance_torque = 0.0
                else:
                    # Agent actively controls closing torque (clamped to gentle negative values)
                    active_close_torque = -abs(cmd_door_torque) * 0.40
                    
                    # Damping heavy resistance if closing too fast
                    if abs(self.door_angular_vel) > self.safe_closing_speed_limit:
                        damping_multiplier = 18.0
                    else:
                        damping_multiplier = 3.5
                        
                    self.applied_door_torque = active_close_torque - damping_multiplier * self.door_angular_vel
                    
                    door_accel = self.applied_door_torque / self.door_inertia
                    self.door_angular_vel += door_accel * sub_dt
                    self.door_angle = max(0.0, min(1.57, self.door_angle + self.door_angular_vel * sub_dt))
                    
                    # Simulated physical obstacle resistance (occurs in 15% of trials, max 1 retry per episode)
                    simulated_obstruction_torque = 0.0
                    if not self.is_completed and 0.20 < self.door_angle < 0.40 and self.retry_count < 1:
                        simulated_obstruction_torque = 3.2
                        
                    # Felt resistance is strictly external obstruction torque, not the robot's own closing effort!
                    self.felt_resistance_torque = simulated_obstruction_torque
                    
                    if self.felt_resistance_torque > self.resistance_threshold:
                        self.safety_reflex_tripped = True
                        self.cooldown_timer = 0
                        self.retry_count += 1
                        
                if self.door_pos_angle() <= 0.05 and not self.is_completed and not self.safety_reflex_tripped:
                    self.active_phase = 4
                    reward += 1000.0 # Milestone bonus for reaching latch zone
                    
            elif self.active_phase == 4:
                self.applied_latch_torque = max(5.0, min(8.0, abs(cmd_door_torque) + 4.0))
                self.applied_door_torque = -self.applied_latch_torque  
                
                door_accel = self.applied_door_torque / self.door_inertia
                self.door_angular_vel += door_accel * sub_dt
                self.door_angle = max(0.0, min(1.57, self.door_angle + self.door_angular_vel * sub_dt))
                
                if self.door_angle <= 0.005:
                    self.is_completed = True

        if self.active_phase == 0:
            handle_dist = np.linalg.norm(self.hand_0_pos - np.array([0.72, 0.0, 0.0]))
            proximity_reward = math.exp(-20.0 * handle_dist) * 5.0
            delta_handle = self.handle_angle - prev_handle_angle
            reward += proximity_reward + (delta_handle * 1500.0) - (self.applied_handle_torque * 0.02)
            
        elif self.active_phase == 1:
            delta_door = self.door_angle - prev_door_angle
            reward += (delta_door * 3500.0)
            
        elif self.active_phase == 2:
            delta_pet = self.pet_pos - prev_pet_pos
            reward += (delta_pet * 3500.0)
            
        elif self.active_phase == 3:
            if self.safety_reflex_tripped:
                # Small flat notice for tripping reflex rather than massive penalty
                reward += -2.0
            else:
                delta_close = prev_door_angle - self.door_angle
                speed_deviation = max(0.0, abs(self.door_angular_vel) - self.safe_closing_speed_limit)
                speed_penalty = -50.0 * (speed_deviation ** 2)
                reward += (delta_close * 3500.0) + speed_penalty
                
        elif self.active_phase == 4:
            force_err = abs(self.applied_latch_torque - 6.5)
            reward += 100.0 - (force_err * 15.0)

        if self.is_damaged:
            reward -= 500.0
        elif self.is_completed:
            reward += 3000.0 # Jackpot for full 5-phase completion

        done = (self.current_step >= EPISODE_LENGTH) or self.is_damaged or self.is_completed
        
        next_obs = self._get_observation()
        return next_obs, reward, done, self.is_completed

    def door_pos_angle(self):
        return self.door_angle

    def _get_observation(self):
        get_observations = []
        get_observations.extend(list(self.hand_0_pos))
        get_observations.extend(list(self.hand_0_vel))
        
        get_observations.append(self.handle_angle / self.max_handle_angle)
        get_observations.append(self.handle_vel / 5.0)
        get_observations.append(self.door_angle / 1.57)
        get_observations.append(self.door_angular_vel / 2.0)
        
        get_observations.append(float(self.active_phase) / 4.0)
        get_observations.append(self.felt_resistance_torque / self.handle_damage_torque)
        get_observations.append(float(self.safety_reflex_tripped))
        get_observations.append(self.pet_pos / 0.60)
        
        get_observations.append(self.applied_handle_torque / self.handle_damage_torque)
        get_observations.append(self.applied_door_torque / 15.0)
        
        get_observations.extend([0.0] * 8)
        return np.array(get_observations, dtype=np.float32)

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
    def __init__(self, state_dim=24, action_dim=5, hidden_dim=64, lr=3e-4, gamma=0.99, K_epochs=3, eps_clip=0.2):
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
        
        if len(returns) > 1:
            advantages = returns - old_values.squeeze(-1)         
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = returns - old_values.squeeze(-1)
        
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

async def run_pet_pipeline():
    print("==================================================================")
    print("      CHRISTENSON ROBOTICS - PORTABLE PET-SAFE DOOR ENTRY GYM     ")
    print("==================================================================")
    print("[Pipeline] Mode: INFINITE TRAINING (Press Ctrl + C to stop)")
    print("[Pipeline] Initializing bimanual GRU Actor-Critic network...")
    print("[Pipeline] Unlatching handle and rotational door inertia: Loaded.")
    print("[Pipeline] Slow-closing, resistance-sensing safety reflex: ACTIVE.")
    
    env = PetInsideEnvironment()
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
        writer.writerow(["episode", "final_completion_status", "peak_resistance_torque_nm", "retry_attempts_count", "reward", "steps_per_sec"])
        
    print(f"[Telemetry Logging] Telemetry initialized at: {TELEMETRY_FILE}\n")
    
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
            max_resistance = 0.0
            ep_start_time = time.time()
            step_count = 0
            
            while not done:
                step_count += 1
                state_t = torch.FloatTensor(state).unsqueeze(0)
                
                action, log_prob, val, next_hx = agent.select_action(state_t, hx_state)
                action_np = action.cpu().numpy().squeeze(0)
                next_state, reward, done, completed_status = env.step(action_np, dt=0.032)
                
                max_resistance = max(max_resistance, env.felt_resistance_torque)
                
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
            print(f"  - Pet Inside Status: {bool(completed_flag)} | Peak Resistance: {max_resistance:.2f}N*m | Retries: {env.retry_count}")
            print(f"  - Episode Reward:    {episode_reward:7.2f} (Rolling Mean: {rolling_reward:7.2f})")
            print("  ------------------------------------------------------------------")
            
            with open(TELEMETRY_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ep, completed_flag, f"{max_resistance:.3f}", env.retry_count, f"{episode_reward:.2f}", f"{sps:.1f}"])
                
            if ep % update_batch_episodes == 0:
                agent.update()
                torch.save(agent.policy.state_dict(), CHECKPOINT_FILE)
            
    except KeyboardInterrupt:
        torch.save(agent.policy.state_dict(), CHECKPOINT_FILE)
        print(f"\n[Checkpoint] Saved trained policy weights to {CHECKPOINT_FILE}")
        print("[Pipeline] Pet gym infinite training paused cleanly by user (Ctrl + C). Telemetry files closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_pet_pipeline())
    except KeyboardInterrupt:
        pass
