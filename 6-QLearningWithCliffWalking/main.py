import gym
import numpy as np

# Créer l'environnement
env = gym.make('CliffWalking-v0', render_mode="human")

# Initialisation des paramètres
n_actions = env.action_space.n
n_states = env.observation_space.n
Q = np.zeros((n_states, n_actions))  # Table Q initialisée à 0
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de réduction
epsilon = 1.0  # Taux d'exploration initial
epsilon_decay = 0.995  # Facteur de réduction de epsilon
epsilon_min = 0.1  # Valeur minimale de epsilon

# Nombre d'épisodes
n_episodes = 1

# Boucle d'apprentissage
for episode in range(n_episodes):
    state = env.reset()[0]  # Réinitialiser l'environnement et obtenir l'état initial
    done = False

    while not done:
        # Choisir l'action avec epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)  # Exploration (action aléatoire)
        else:
            action = np.argmax(Q[state, :])  # Exploitation (meilleure action selon Q)

        # Exécuter l'action dans l'environnement
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Mettre à jour la Q-value
        best_next_action = np.argmax(Q[next_state, :])  # Trouver la meilleure action suivante
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

        print(f"Épisode {episode + 1}, Étape {state} -> Action choisie: {action}, Récompense: {reward}, Nouvelle Q-value: {Q[state, action]}")

        # Passer à l'état suivant
        state = next_state

    # Diminuer epsilon pour réduire l'exploration au fur et à mesure des épisodes
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Optionnel : Afficher l'épisode tous les 100 épisodes
    if (episode + 1) % 100 == 0:
        print(f"Épisode {episode + 1}/{n_episodes} terminé")

# Test de l'agent après apprentissage
state = env.reset()[0]
done = False
while not done:
    action = np.argmax(Q[state, :])  # Choisir la meilleure action selon Q
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    env.render()  # Afficher l'environnement
    
print('Matrice finale')
print(Q)