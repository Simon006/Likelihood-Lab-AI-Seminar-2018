import gym


class CartPoleEnv:
    def __init__(self, agent, game_epoch):
        # construct gym cart pole environment
        self._env = gym.make('CartPole-v0')
        self._env = self._env.unwrapped

        # define the reinforcement learning agent
        self._agent = agent

        # number of games to play
        self._game_epoch = game_epoch

    def run(self):
        for e in range(self._game_epoch):
            # receive initial observation
            observation_current = self._env.reset()

            # performances of this game epoch
            total_step = 0
            total_reward = 0

            while True:
                # render the game to screen
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

                # update game information
                total_step += 1
                total_reward += reward

                # if the game is finished, we reset the game to restart.
                # if the game is not finished, we keep on playing in the current game.
                if is_done:
                    print('>>> the {0}th game is over with {1} total steps and {2} total rewards.'.format(e+1,
                                                                                                          total_step,
                                                                                                          total_reward))
                    break
                else:
                    observation_current = observation_next
