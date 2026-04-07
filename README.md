# 🚀 Lunar Lander — Solved with (D)DQN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow%2FKeras-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Gymnasium-LunarLander--v3-green" />
  <img src="https://img.shields.io/badge/Algorithm-DDQN-purple" />
</p>

<p align="center">
  <b>Trained Agent Landing</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Untrained Agent Landing</b>
</p>
<p align="center">
  <video src="video/agentm5.mp4" width="45%" controls autoplay loop muted></video>
  &nbsp;&nbsp;
  <video src="video/agentuntrained.mp4" width="45%" controls autoplay loop muted></video>
</p>

---

## 📋 Table of Contents

- [Problem Definition](#-problem-definition)
- [Background & Theory](#-background--theory)
- [Method](#-method)
- [Architecture](#-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Training & Usage](#-training--usage)
- [Implementation Notes](#-implementation-notes)
- [References](#-references)

---

## 🌕 Problem Definition

The **Lunar Lander** environment ([Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/)) simulates landing a small spacecraft on the moon surface. The goal is to navigate the lander between two flags at coordinates `(0, 0)` without crashing.

### State Space (8-dimensional)

| Index | Description |
|-------|-------------|
| 0, 1 | x, y coordinates of the lander |
| 2, 3 | Linear velocities in x, y |
| 4 | Lander angle |
| 5 | Angular velocity |
| 6, 7 | Boolean — whether each leg is in contact with the ground |

### Action Space (Discrete, 4 actions)

| Action | Description |
|--------|-------------|
| `0` | Do nothing |
| `1` | Fire left orientation engine |
| `2` | Fire main engine |
| `3` | Fire right orientation engine |

### Reward Structure

- ✅ **+100** for landing safely
- ❌ **−100** for crashing
- 📍 Increases/decreases based on proximity to landing pad
- 🐢 Increases/decreases based on lower/higher speed
- 📐 Decreases when the lander is tilted
- 🦵 **+10** per leg in contact with the ground
- 🔥 **−0.03** per frame a side engine fires
- 🔥 **−0.3** per frame the main engine fires

> An episode is considered **solved** when it achieves a score ≥ 200 points (averaged over 100 consecutive episodes).

---

## 📚 Background & Theory

The Lunar Lander features a large, continuous state space — making tabular Q-learning infeasible. The solution leverages **Deep Q-Networks (DQN)**, which use a neural network as a function approximator for the Q-value instead of a lookup table.

### Why not simpler methods?
- **Dynamic Programming** — requires perfect environment knowledge (not available here)
- **Monte Carlo** — requires complete episodes; slow convergence and high memory usage
- **Tabular Q-learning** — infeasible for continuous state spaces

### DQN vs DDQN

Standard **DQN** uses a single neural network for both action selection and target Q-value estimation. This introduces a **maximisation bias** — the agent overestimates Q-values because the same network selects and evaluates actions simultaneously.

**Double DQN (DDQN)** solves this by using two networks:
- **Online network** — selects the best action
- **Target network** — evaluates that action's Q-value

The target network is periodically synced with the online network (every N steps), keeping target values more stationary and improving training stability.

---

## 🧠 Method

Both DQN and DDQN were implemented and compared. The core components are:

### Experience Replay Buffer
A fixed-size memory buffer stores past `(state, action, reward, next_state, done)` transitions. At each training step, a random mini-batch is sampled, breaking temporal correlations and stabilising learning.

### Training Loop (per episode step)
1. Agent observes the current state
2. Chooses an action via **ε-greedy policy** (explore vs exploit)
3. Executes the action, receives reward & next state
4. Stores the transition in the replay buffer
5. Every 4 steps, samples a mini-batch and trains the online network
6. Every N steps, copies online network weights to target network
7. Decays ε over time (reduces exploration as agent improves)

### Key Hyperparameters (Best Model — m5)

| Parameter | Value |
|-----------|-------|
| Algorithm | DDQN |
| ML Library | PyTorch |
| Memory Size | 500,000 transitions |
| Neural Network | 2 hidden layers × 128 neurons |
| Batch Size | 128 |
| Episodes | 1,500 |
| Learn every N steps | 4 |
| Target network sync | Every 100 steps |
| Learning Rate | 0.001 |
| Discount (γ) | 0.99 |
| Epsilon decay | 0.996 |
| Epsilon min | 0.01 |

---

## 🏗️ Architecture
