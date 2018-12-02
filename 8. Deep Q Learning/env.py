import gym


class Env:
    def __init__(self):
        pass


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    env.render()
    env.close()
