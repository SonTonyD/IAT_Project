from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

def main():

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()

    gamma = 0.8
    alpha = 0.1
    n_episodes = 5
    max_iter = 1000
    epsilon = 0.9


    controller = RandomAgent(8, 12, 2, game.na, game, gamma, alpha, n_episodes, max_iter, epsilon)
    # controller.learn()
    # controller.saveQ("Q_table_0.npy")
    controller.loadQ("Q_table_0.npy")
 
    state = game.reset()
    while True:

        try:
            action = controller.select_action(state)
            state, reward, is_done = game.step(action)
            sleep(0.0001)
        except IndexError:
            print(game.score_val)
            quit()
    

if __name__ == '__main__' :
    main()