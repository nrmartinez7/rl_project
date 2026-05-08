# RL Final Project

This project explores reinforcement learning using OpenAI Gymnasium environments. We start with the Taxi environment using Q-Learning and transition to CartPole for more advanced reinforcement learning techniques. 

## Create the virtual environment

```bash
python3 -m venv venv
```

Activate the environment:
### Mac/Linux
```bash
source venv/bin/activate
```
### Windows
```bash
venv\Scripts\activate
```
---

## Install dependencies

```bash
pip install -r requirements.txt
```
```bash
pip install pygame
```

---

## Running the files

### Taxi
Run a random Taxi agent:

```bash
python taxi/taxi_random.py
```

Run Q-learning on Taxi:

```bash
python taxi/taxi_qlearning.py
```

### CartPole
Run a random Cartpole agent:

```bash
python cartpole/cartpole_random.py
```

Run DQN on CartPole:

```bash
python cartpole/dqn_cartpole.py
```

---
