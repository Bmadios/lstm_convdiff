import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from model_lstm import *
from plotting import *
import optuna
import time

# Début du chronomètre
start_time = time.time()

"""
python optimisation_Optuna.py

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device

# Supposons que data soit un tableau numpy de forme (N, 4) où N est le nombre d'échantillons
# et chaque ligne est un quadruplet (t, x, y, u)
file_path = "/notebooks/solution_u_implicit_data.csv"
data = pd.read_csv(file_path)

def truncate_two_decimals(value):
    return int(value * 100) / 100.0

def truncate_three_decimals(value):
    return int(value * 1000) / 1000.0

u_at_0_003 = data[data["temps"] == 0.003]["u"].values

# Étape 2: Remplacez les valeurs u aux timesteps 0, 0.001 et 0.002 par cette valeur
#for t in [0, 0.001, 0.002]:
    #data.loc[data["temps"] == t, "u"] = u_at_0_003

data["x"] = data["x"].apply(truncate_two_decimals)
data["y"] = data["y"].apply(truncate_two_decimals)
data["temps"] = data["temps"].apply(truncate_three_decimals)
data["u"] = data["u"].apply(truncate_three_decimals)

# Séparation des entrées et des cibles en utilisant .iloc
inputs = data.iloc[:, :-1].values  # Prend tout sauf la dernière colonne
targets = data.iloc[:, -1].values  # Prend seulement la dernière colonne



n = len(inputs)
idx_train = int(n * 0.4)
#idx_val = idx_train + int(n * 0.2)


# Fonction pour créer des séquences
def create_sequences(data, targets, sequence_length):
    X, Y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(targets[i+sequence_length])
    return np.array(X), np.array(Y)

def create_sequences_with_full_intervals(data, targets, sequence_length, interval_length=1681):
    X, Y = [], []
    for i in range(0, len(data) - sequence_length, interval_length):
        for j in range(i, i + interval_length - sequence_length):
            X.append(data[j:j+sequence_length])
            Y.append(targets[j+sequence_length])
    return np.array(X), np.array(Y)

# Création de séquences pour chaque ensemble
# Création de séquences
sequence_length = 10



X, Y = create_sequences(inputs, targets, sequence_length)
#X_val, Y_val = create_sequences_with_full_intervals(X_val, Y_val, sequence_length)
#X_test, Y_test = create_sequences_with_full_intervals(X_test, Y_test, sequence_length)


X_train = X[:idx_train]
Y_train = Y[:idx_train]

X_test = X[idx_train:]
Y_test = Y[idx_train:]

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)


# Déplacer les tensors vers le GPU si disponible
if torch.cuda.is_available():
    X_train_tensor = X_train_tensor.cuda()
    Y_train_tensor = Y_train_tensor.cuda()
    X_val_tensor = X_val_tensor.cuda()
    Y_val_tensor = Y_val_tensor.cuda()
    X_test_tensor = X_test_tensor.cuda()
    Y_test_tensor = Y_test_tensor.cuda()

input_dim = 3  # (t, x, y)
hidden_dim = 96
layer_dim = 1 # CASE 1 (H =1) CASE 02 (H = 2)
output_dim = 1

# Fonction d'objectif pour Optuna
def objective(trial):
    # Hyperparamètres à optimiser
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 76, 96])
    layer_dim = trial.suggest_int("layer_dim", 1, 3)
    
    # Initialisation du modèle avec les hyperparamètres proposés
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    #model.load_state_dict(torch.load('model_path_test.pth'))
    #model.eval()  # Mettez le modèle en mode évaluation
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement avec les hyperparamètres actuels
    for epoch in range(1000):  # Utilisez un plus petit nombre d'époques pour l'optimisation
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze(1)
        loss = criterion(outputs, Y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        val_outputs = model(X_val_tensor).squeeze(1)
        val_loss = criterion(val_outputs, Y_val_tensor)
        
        if val_loss.item() < 2.5e-8:
            break

    # Le but est de minimiser la perte de validation
    return val_loss.item()

# Création de l'étude Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # Nombre d'essais à réaliser

# Affichage des meilleurs hyperparamètres
print("Meilleurs hyperparamètres: ", study.best_params)

end_time = time.time()

execution_time = end_time - start_time

# Convertir en heures, minutes, secondes
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60

# Écrire dans un fichier
with open("temps_execution.txt", "w") as file:
    file.write(f"Temps d'exécution: {hours} heure(s), {minutes} minute(s) et {seconds:.2f} seconde(s)\n")

