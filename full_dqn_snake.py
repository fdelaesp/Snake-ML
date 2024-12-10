# full_dqn_snake.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os
import numpy as np

# Define Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# File paths
MODEL_PATH = 'best_model.pth'
BEST_SCORE_PATH = 'best_score.txt'

# Training Parameters
MAX_GAMES = 500
MAX_NO_IMPROVEMENT = 100  # Stop after 100 consecutive games without improvement in mean score
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
TARGET_UPDATE_FREQ = 10  # Update target network every 10 games
MEMORY_CAPACITY = 100_000
PRIORITIZED_REPLAY_ALPHA = 0.6
PRIORITIZED_REPLAY_BETA_START = 0.4
PRIORITIZED_REPLAY_BETA_FRAMES = 1000  # Anneal beta to 1 over these frames

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience tuple
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(*args))
        else:
            self.buffer[self.pos] = Experience(*args)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = Experience(*zip(*samples))

        states = torch.tensor(batch.state, dtype=torch.float).to(device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(batch.reward, dtype=torch.float).unsqueeze(1).to(device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float).to(device)
        dones = torch.tensor(batch.done, dtype=torch.bool).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1).to(device)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            # Extract scalar value from prio before assignment to fix DeprecationWarning
            if isinstance(prio, torch.Tensor):
                prio = prio.item()
            elif isinstance(prio, np.ndarray):
                prio = prio.item()
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# Snake Game Environment
class SnakeGame:
    def __init__(self, render=False, fps=60):
        """
        Initialize the Snake game environment.

        Args:
            render (bool): Whether to render the game visually.
            fps (int): Frames per second for the game. Lower values slow down the game.
        """
        self.render_mode = render
        self.width = 400
        self.height = 400
        self.block_size = 20
        self.fps = fps  # Add fps parameter
        self.reset()

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake Game')
            self.clock = pygame.time.Clock()

    def reset(self):
        self.x = self.width // 2
        self.y = self.height // 2
        self.snake = [[self.x, self.y]]
        self.food = self.place_food()
        self.direction = RIGHT
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()

    def place_food(self):
        while True:
            food = [random.randrange(0, self.width, self.block_size),
                    random.randrange(0, self.height, self.block_size)]
            if food not in self.snake:
                return food

    def step(self, action):
        self.frame_iteration += 1
        # Update direction
        if action == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif action == RIGHT and self.direction != LEFT:
            self.direction = RIGHT
        elif action == UP and self.direction != DOWN:
            self.direction = UP
        elif action == DOWN and self.direction != UP:
            self.direction = DOWN

        # Move the snake
        if self.direction == UP:
            self.y -= self.block_size
        elif self.direction == DOWN:
            self.y += self.block_size
        elif self.direction == LEFT:
            self.x -= self.block_size
        elif self.direction == RIGHT:
            self.x += self.block_size

        # Check for collisions
        done = False
        reward = 0
        if self.x < 0 or self.x >= self.width or self.y < 0 or self.y >= self.height:
            done = True
            reward = -10
            return self.get_state(), reward, done
        if [self.x, self.y] in self.snake:
            done = True
            reward = -10
            return self.get_state(), reward, done

        # Update snake
        self.snake.append([self.x, self.y])

        # Check if food eaten
        if [self.x, self.y] == self.food:
            self.score += 1
            reward = 10
            self.food = self.place_food()
        else:
            self.snake.pop(0)
            reward = -0.1  # Slight penalty for not eating to encourage faster food acquisition

        # Prevent infinite loops
        if self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10

        if self.render_mode:
            self.render()

        return self.get_state(), reward, done

    def get_state(self):
        snake_head = self.snake[-1]
        food = self.food

        # Calculate normalized distances to food
        distance_to_food = [
            (food[0] - snake_head[0]) / self.width,
            (food[1] - snake_head[1]) / self.height
        ]

        # Current direction (one-hot)
        direction = [
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(self.direction == RIGHT)
        ]

        # Danger indicators: straight, right, left
        danger = self.get_danger()

        state = distance_to_food + direction + danger  # Total 2 + 4 + 3 = 9
        return state

    def get_danger(self):
        dangers = [0, 0, 0]

        # Define the relative directions based on current movement
        if self.direction == UP:
            front = [self.x, self.y - self.block_size]
            right = [self.x + self.block_size, self.y]
            left = [self.x - self.block_size, self.y]
        elif self.direction == DOWN:
            front = [self.x, self.y + self.block_size]
            right = [self.x - self.block_size, self.y]
            left = [self.x + self.block_size, self.y]
        elif self.direction == LEFT:
            front = [self.x - self.block_size, self.y]
            right = [self.x, self.y + self.block_size]
            left = [self.x, self.y - self.block_size]
        elif self.direction == RIGHT:
            front = [self.x + self.block_size, self.y]
            right = [self.x, self.y - self.block_size]
            left = [self.x, self.y + self.block_size]

        # Check dangers
        dangers_positions = [front, right, left]
        for i, pos in enumerate(dangers_positions):
            x, y = pos
            if x < 0 or x >= self.width or y < 0 or y >= self.height or pos in self.snake:
                dangers[i] = 1

        return dangers

    def render(self):
        self.screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (*segment, self.block_size, self.block_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (*self.food, self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(self.fps)  # Use the fps parameter

    def close(self):
        pygame.quit()

# Neural Network for DQN
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name=MODEL_PATH):
        torch.save(self.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load(self, file_name=MODEL_PATH):
        if os.path.exists(file_name):
            try:
                self.load_state_dict(torch.load(file_name, map_location=device))
                self.to(device)
                self.eval()
                print(f"Loaded model from {file_name}")
            except RuntimeError as e:
                print(f"Failed to load model from {file_name}: {e}")
                print("Starting with a fresh model.")
        else:
            print(f"No model found at {file_name}, starting fresh.")

# DQN Agent with Double DQN and Prioritized Experience Replay
class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, lr=LR, gamma=GAMMA,
                 memory_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
                 alpha=PRIORITIZED_REPLAY_ALPHA):
        self.n_games = 0
        self.epsilon = 1.0  # Start with high exploration
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = PRIORITIZED_REPLAY_BETA_START
        self.beta_increment = (1.0 - PRIORITIZED_REPLAY_BETA_START) / PRIORITIZED_REPLAY_BETA_FRAMES

        self.memory = PrioritizedReplayBuffer(memory_size, alpha=self.alpha)

        self.model = Linear_QNet(input_size, hidden_size, output_size).to(device)
        self.model.load()  # Load existing model if available

        self.target_model = Linear_QNet(input_size, hidden_size, output_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')  # We'll apply weights manually

        # Load best score if exists
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH, 'r') as f:
                self.best_score = float(f.read())
            print(f"Loaded best score: {self.best_score}")
        else:
            self.best_score = 0

    def get_action(self, state, training=True):
        """
        Returns an action based on the current state using an epsilon-greedy policy.

        Args:
            state (list): Current state of the game.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: Action to take.
        """
        if training:
            self.epsilon = max(0.05, self.epsilon - 0.001)  # Exponential decay
            if random.random() < self.epsilon:
                move = random.randint(0, 3)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = self.model(state_tensor)
                move = torch.argmax(prediction).item()
        else:
            # Deterministic action for visualization
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
        return move

    def remember(self, state, action, reward, next_state, done):
        """
        Stores experiences in memory.

        Args:
            state (list): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (list): Next state after action.
            done (bool): Whether the game is done.
        """
        self.memory.push(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Trains the model on a single step.

        Args:
            state (list): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (list): Next state after action.
            done (bool): Whether the game is done.
        """
        self.remember(state, action, reward, next_state, done)
        self.learn()

    def train_long_memory(self):
        """
        Trains the model on a batch of experiences.
        """
        self.learn()

    def learn(self):
        """
        Performs a learning step: samples a batch, computes targets, calculates loss, and updates the network.
        """
        if len(self.memory) < self.batch_size:
            return

        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, self.beta)

        # Current Q values
        current_q = self.model(states).gather(1, actions)

        # Double DQN logic
        with torch.no_grad():
            # Select actions with main network
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            # Evaluate Q-values with target network
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (~dones))

        # Compute TD errors for prioritized replay
        td_errors = target_q - current_q
        loss = (self.loss_fn(current_q, target_q) * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities)

    def update_target_network(self):
        """
        Updates the target network with the weights from the main network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def update_best_score(self, score):
        """
        Updates the best score and saves it.

        Args:
            score (float): The score to compare against the best score.
        """
        if score > self.best_score:
            self.best_score = score
            with open(BEST_SCORE_PATH, 'w') as f:
                f.write(str(score))
            self.model.save()
            print(f'New Record! Score: {score}')

# Training Loop
def train():
    scores = []
    mean_scores = []
    total_score = 0
    no_improvement_games = 0
    best_mean_score = 0  # To track improvements in mean score
    agent = DQNAgent(input_size=9, hidden_size=256, output_size=4)
    game = SnakeGame(render=False)  # Training without rendering

    for _ in range(MAX_GAMES):
        state = game.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done = game.step(action)

            # Train short memory
            agent.train_short_memory(state, action, reward, next_state, done)

            state = next_state
            score += reward

        # Train long memory after each game
        agent.train_long_memory()

        agent.n_games += 1
        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        print(f'Game {agent.n_games} | Score: {score:.2f} | Record: {agent.best_score} | Mean Score: {mean_score:.2f}')

        # Update best score if necessary
        if score > agent.best_score:
            agent.update_best_score(score)

        # Check for improvement in mean score
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            no_improvement_games = 0
        else:
            no_improvement_games += 1

        # Early Stopping
        if no_improvement_games >= MAX_NO_IMPROVEMENT:
            print(f"Early stopping at game {agent.n_games} due to no improvement in mean score for {MAX_NO_IMPROVEMENT} consecutive games.")
            break

        # Update target network periodically
        if agent.n_games % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            print("Updated target network.")

    # Plotting the scores
    plt.figure(figsize=(12, 5))
    plt.title('Training Progress')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Game')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.show()

    # Save the final model if it's the best
    if score > agent.best_score:
        agent.update_best_score(score)

    game.close()

# Visualization Function
def visualize():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Please train the agent first.")
        return

    agent = DQNAgent(input_size=9, hidden_size=256, output_size=4)
    agent.model.load(MODEL_PATH)
    agent.model.eval()
    print("Loaded the best model for visualization.")

    # Set a lower fps for visualization to slow down the snake's movement
    visualization_fps = 10  # Adjust this value as needed (e.g., 5, 10, 15)
    game = SnakeGame(render=True, fps=visualization_fps)
    state = game.reset()
    done = False
    score = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = agent.get_action(state, training=False)
        next_state, reward, done = game.step(action)
        state = next_state
        score += reward

    print(f'Final Score: {score}')
    game.close()

if __name__ == '__main__':
    choice = input("Enter 'train' to train the agent or 'visualize' to see the agent play: ").lower()
    if choice == 'train':
        train()
    elif choice == 'visualize':
        visualize()
    else:
        print("Invalid choice. Please enter 'train' or 'visualize'.")
