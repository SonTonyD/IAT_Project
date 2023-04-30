# Projet IAT 2023 - Algorithme d'apprentissage par renforcement pour Space Invader
Ce projet consiste en l'implémentation d'un algorithme d'apprentissage par renforcement pour jouer au jeu Space Invader en utilisant la bibliothèque Pygame.

## Méthode choisie
Dans ce projet, la méthode Q-learning a été choisie pour l'apprentissage par renforcement. Cette méthode utilise une fonction de valeur Q pour estimer la récompense future d'une action donnée dans un état donné.

## Installation
1. Récupérer le projet
```bash
git clone https://github.com/SonTonyD/IAT_Project.git
cd IAT-projet
```

2. Installer les dépendances
```bash
pip3 install -r requirements.txt
```

3. Configurer les hyperparamètres
Remplir le fichier texte : hyperparameter_test_set.txt en respectant le format suivant : 
<display: True or False> <gamma> <alpha> <n_episodes> <max_iter> <epsilon> <model_name> <mode>

Il existe 2 modes: 
- le mode "train" : le programme va se charger d'entrainer un agent avec les paramètres renseignés et , à la fin, l'enregister dans un fichier <model_name> avec l'extension .npy

- le mode "sim" : le programme va lancer une simulation (1 partie) en chargeant le modèle indiqué. Les résultats seront sauvegardés dans un fichier results.csv.

Il est d'ailleurs tout à fait possible de lancer plusieurs simulations d'affiler en renseignant plusieurs lignes dans le fichier : hyperparameter_test_set.txt


4. Lancer le programme
```bash
./create-agent.sh
```

## Le sujet du projet

Un fichier `subject.pdf` contenant les instructions du projet est disponible.

## Auteurs
Ce programme a été écrit par Hajj Christian, Sami Hani, Brosset Ilona, Dinh Son-Tony
