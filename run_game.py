from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

def main():

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()

    gamma = 0.8
    alpha = 0.1
    n_episodes = 10
    max_iter = 5000
    epsilon = 0.9


    controller = RandomAgent(160, 120, 2, game.na, game, gamma, alpha, n_episodes, max_iter, epsilon)
    controller.learn()
 
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()