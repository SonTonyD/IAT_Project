# from time import sleep
# from game.SpaceInvaders import SpaceInvaders
# from controller.keyboard import KeyboardController
# from controller.random_agent import RandomAgent

# def main():

#     game = SpaceInvaders(display=True)
#     #controller = KeyboardController()

#     gamma = 0.8
#     alpha = 0.1
#     n_episodes = 5
#     max_iter = 2000
#     epsilon = 0.9


#     controller = RandomAgent(8, 12, 2, game.na, game, gamma, alpha, n_episodes, max_iter, epsilon)
    
#     model_name = "Q_table_0.npy"

#     controller.learn()
#     controller.saveQ("Q_table_0.npy")

#     model_name = "Q_table_0.npy"
#     controller.loadQ(model_name)
 
#     state = game.reset()
#     while True:

#         try:
#             action = controller.select_action(state)
#             state, reward, is_done = game.step(action)
#             sleep(0.0001)
#         except IndexError:
#             print(game.score_val)
#             controller.append_to_csv("results.csv", [model_name, gamma, alpha, n_episodes, max_iter, epsilon, game.score_val])
#             quit()
    

# if __name__ == '__main__' :
#     main()


import sys
from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

def train(controller, model_name):
    controller.learn()
    controller.saveQ(model_name)

def simulation(controller, model_name, game):
    controller.loadQ(model_name)
    state = game.reset()
    while True:
        try:
            action = controller.select_action(state)
            state, reward, is_done = game.step(action)
            sleep(0.0001)
        except IndexError:
            print(game.score_val)
            controller.append_to_csv("results.csv", [model_name, gamma, alpha, n_episodes, max_iter, epsilon, game.score_val])
            quit()


def main(display, gamma, alpha, n_episodes, max_iter, epsilon, model_name, mode):
    game = SpaceInvaders(display=display)
    gamma = float(gamma)
    alpha = float(alpha)
    n_episodes = int(n_episodes)
    max_iter = int(max_iter)
    epsilon = float(epsilon)
    model_name = str("Q_table_0.npy")
    mode = str(mode)


    controller = RandomAgent(8, 12, 2, game.na, game, gamma, alpha, n_episodes, max_iter, epsilon)    
    if mode == "train":
        print("TRAINING AGENT")
        train(controller, model_name)
    
    if mode == "sim":
        print("SIMULATION RUNNING")
        simulation(controller, model_name, game)
        







if __name__ == '__main__' :
    if len(sys.argv) != 9:
        print("Usage: python3 run_game.py <display: True or False> <gamma> <alpha> <n_episodes> <max_iter> <epsilon> <model_name> <mode>")
        quit()

    display = True if sys.argv[1].lower() == 'true' else False
    gamma = sys.argv[2]
    alpha = sys.argv[3]
    n_episodes = sys.argv[4]
    max_iter = sys.argv[5]
    epsilon = sys.argv[6]
    model_name = sys.argv[7]
    mode = sys.argv[8]

    main(display, gamma, alpha, n_episodes, max_iter, epsilon, model_name, mode)
