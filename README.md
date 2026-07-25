Christenson Robotics - Haptic Dexterity Wrapper Core

Welcome to the public wrapper repository for Christenson Robotics. This open-source repository houses the visualizer, standard ROS2 bridge nodes, and local verification gyms designed to let roboticists, researchers, and hobbyists stream real-time visual-tactile sensory vectors from their hardware directly to our 3D Interactive Web HUD.

🚀 The Mission: The OS of Haptic Dexterity

Most humanoid robotics systems today are structurally "numb." They rely heavily on visual cameras to reach for objects, but lack the microsecond active tactile reflexes necessary to handle delicate, wet, or highly deformable items (like washing fragile wine glasses or carrying flexible silicone ice trays).

Christenson Robotics is building a pre-trained, recurrent tactile-reflex control system.

┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Your Physical Robot Arm     │       │   Christenson Robotics Brain    │
│  (Unitree, Fourier, UR5, etc.)  │       │     (Proprietary Core Layer)    │
└────────────────┬────────────────┘       └────────────────▲────────────────┘
                 │                                         │
                 │  Streams tactile & joint states         │  Returns optimized
                 ▼                                         │  torque commands
┌──────────────────────────────────────────────────────────┴────────────────┘
│            Open-Source ROS2 Bridge Node & Web HUD Visualizer            │
│                       (This AGPLv3 Repository)                           │
└───────────────────────────────────────────────────────────────────────────┘


🛠️ Onboarding & Installation

To run the local developer simulation gyms and compile the haptic telemetry HUD on your machine:

Clone the repository:

git clone https://github.com/spencerc89/christenson_robotics.git
cd christenson_robotics


Set up your environment and dependencies:

pip install -r requirements.txt
pip install warp-lang


Launch the Dishwasher Loading Gym:

python sim/dishwasher_loading_sim.py


Launch the Pet-Safe Door Entry Gym:

python sim/let_pets_inside_sim.py


Launch the Countertop Wiping Gym:

python sim/wipe_counter_sim.py


📜 Licensing & Dual-Licensing Strategy

This repository's wrappers and visualizers are licensed under the GNU Affero General Public License v3 (AGPLv3).

If you are a commercial robotics manufacturer (OEM) or plan to package these wrappers inside proprietary commercial products without exposing your own code under copyleft provisions, you must contact us to purchase a Commercial/Enterprise Tier license exemption.

For pricing, priority SLAs, and custom recurrent model training, contact our team at licensing@christensonrobotics.com.
