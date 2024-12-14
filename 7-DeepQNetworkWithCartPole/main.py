import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

env = gym.make('CartPole-v1', render_mode="human")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24) # Première couche (entrée -> 24 neurones)
        self.fc2 = nn.Linear(24, 24) # Deuxième couche (24 -> 24 neurones)
        self.fc3 = nn.Linear(24, action_dim) # Dernière couche (24 -> actions possibles)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Activation ReLU sur la première couche
        x = torch.relu(self.fc2(x)) # Activation ReLU sur la deuxième couche
        return self.fc3(x) # Prédiction de la Q-value pour chaque action

class ReplayBuffer: # Stocker les transitions pour être réutilisées pendant l'entrainement
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done): # Ajouter dans la mémoire
        self.buffer.append(((state, action, reward, next_state, done)))
    
    def sample(self, batch_size): # Retirer un échantillon aléatoire de la mémoire
        return random.sample(self.buffer, batch_size)
    
    def size(self): # Retourne la taille
        return len(self.buffer)

state_dim = env.observation_space.shape[0] # Taille de l'état (4 pour CartPole)
action_dim = env.action_space.n # Nb d'actions possible (2 pour CartPole)

# Créer le modèle DQN
model = DQN(state_dim, action_dim)
target_model = DQN(state_dim, action_dim)
target_model.load_state_dict(model.state_dict()) # Initialiser le poids du modèle cible

# Optimiseur et critère de perte
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Paramètres de l'agent
gamma = 0.99 # Facteur de réduction
epsilon = 0.1 # Taux d'exploration
epsilon_decay = 0.995 # Décroissance de l'exploration
epsilon_min = 0.01 # Exploration minimale
batch_size = 32 # Taille du batch pour l'entraînement
replay_buffer = ReplayBuffer(10000) # Capacité de mémoire du tampon
target_update_freq = 10

# Entraînement de l'agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()[0] # Reset l'environnement
    done = False
    total_reward = 0

    while not done:
        # Selection de l'action
        if random.random() < epsilon:
            action = random.choice(range(action_dim)) # Exploration (choix d'une action en aléatoire)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = torch.argmax(model(state_tensor)).item() # Exploitation (choix de la meilleure action)
        
        # Effectuer l'action et observer la récompense et le nouvel état
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Stocker l'expérience dans la mémoire
        replay_buffer.push(state, action, reward, next_state, done)

        # Si la mémoire est suffisamment remplie et entraine le modèle
        if replay_buffer.size() >= batch_size:
            # Echantilloner un mini-batch
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Calcul des QValues cibles
            next_q_value = target_model(next_states).max(1)[0]
            target_q_value = rewards + gamma * next_q_value * (1 - dones)

            # Prédiction des QValues pour le mini_batch
            current_q_value = model(states).gather(1, actions.unsqueeze(1).long())

            # Calcul de la perte
            loss = criterion(current_q_value, target_q_value.unsqueeze(1))
            
            # Mise à jour des poids
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # MAJ de l'epsilone (réduction de l'exploration)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # MAJ du modèle cible
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

    print(f"Épisode {episode + 1}/{num_episodes}, Récompense totale: {total_reward}, Epsilon: {epsilon}")

# Teste de l'agent après entraînement
print("Test final")
state = env.reset()[0]
done = False
while True:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.argmax(model(state_tensor)).item()
    state, reward, done, _, _ = env.step(action)
    env.render()

    if done:
        print("Episode terminé, Réinitialisation de l'environnement...")
        state = env.reset()[0]