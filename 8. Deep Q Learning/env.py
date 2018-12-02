import gym
import numpy as np
import matplotlib.pyplot as plt


class CartPoleEnv:
    def __init__(self, agent, game_epoch, is_render_image, is_verbose):
        # construct gym cart pole environment
        self._env = gym.make('CartPole-v0')
        self._env = self._env.unwrapped

        # define the reinforcement learning agent
        self._agent = agent

        # number of games to play
        self._game_epoch = game_epoch

        # display the game on the screen
        self._is_render_image = is_render_image

        # display the reward information of each game
        self._is_verbose = is_verbose

    def run(self):
        # record each game's reward
        reward_list = []

        # play No.e game
        for e in range(self._game_epoch):

            # print information if we have enough games
            if (len(reward_list) % 1000 == 0) and (len(reward_list) > 0):
                print('> average recent game rewards: ' + str(np.average(reward_list[-1000:])))

            # receive initial observation
            observation_current = self._env.reset()

            # performances of this game epoch
            reward_this_epoch = 0

            # run this game till end
            while True:
                # render the game to screen
                if self._is_render_image:
                    self._env.render()

                # reinforcement learning agent choose which action to take
                action = self._agent.choose_action(observation_current)

                # environment receives the action the agent took
                observation_next, reward, is_done, info = self._env.step(action)

                # store the data for training
                self._agent.store_train_data(observation_current, action, reward, observation_next, is_done)

                # train the agent when data is enough
                if self._agent.have_enough_new_data():
                    self._agent.train()

                # clear the train data if it is too many
                self._agent.clear_excessive_data()

                # update game information
                reward_this_epoch += reward

                # if the game is finished, we reset the game to restart.
                # if the game is not finished, we keep on playing in the current game.
                if is_done:
                    if self._is_verbose:
                        print('> reward for the {0}th game is {1}.'.format(e+1, reward_this_epoch))
                    break
                else:
                    observation_current = observation_next
                    continue

            # record the performance of this game epoch
            reward_list.append(reward_this_epoch)

        # plot the reward list
        plt.plot(reward_list, label='reward', color='lightgray')
        plt.legend(loc=1)
        plt.xlabel('Game Number')
        plt.ylabel('Reward')
        plt.show()
