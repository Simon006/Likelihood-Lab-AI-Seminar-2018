import gym
from dqn import DeepQNet


class CartPoleEnv:
    def __init__(self, agent, epoch):
        # construct gym cart pole environment
        self._env = gym.make('CartPole-v0')
        self._env = self._env.unwrapped

        # define the reinforcement learning agent
        self._agent = agent

        # number of games to play
        self._epoch = epoch

    def run(self):
        for e in range(self._epoch):
            # receive initial observation
            observation_current = self._env.reset()

            while True:
                # render the game to screen
                self._env.render()

                # reinforcement learning agent choose which action to take
                action = self._agent.choose_action(observation_current)

                # environment receives the action the agent took
                observation_next, reward, done, info = self._env.step(action)

                # store the data for training
                self._agent.store_train_data(observation_current, action, reward, observation_next)

                # train the agent when data is enough
                if self._agent.have_enough_data():
                    self._agent.train()

                # if the game is finished, we reset the game to restart.
                # if the game is not finished, we keep on playing in the current game.
                if done:
                    break
                else:
                    observation_current = observation_next


if __name__ == '__main__':
    dqn = DeepQNet()
    env = CartPoleEnv(agent=dqn, epoch=1000)
    env.run()
