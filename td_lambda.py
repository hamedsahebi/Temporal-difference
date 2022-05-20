import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gym import wrappers
import datetime
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class FeatureTransformer:

    def __init__(self, env, n_components=500):

        observation_examples = np.array(
            [env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
        ])

        feature_examples = featurizer.fit_transform(
            scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):

        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class BaseModel:
    def __init__(self, D):
        self.w = np.random.randn(D)/np.sqrt(D)

    def partial_fit(self, input_, target, eligibility, lr=10e-3):
        self.w += lr*(target - input_.dot(self.w)) * eligibility

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)


# One base model for each action

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))

        for i in range(env.action_space.n):
            model = BaseModel(D)
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X) for m in self.models])

    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.eligibilities[a] *= gamma * lambda_

        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one(model, eps, gamma, lambda_):

    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)

        # Update the model
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G, gamma, lambda_)

        totalreward += reward
        iters += 1

    return totalreward


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)

    Z = np.apply_along_axis(
        lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-to-go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):t+1].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    lambda_ = 0.7

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    total_rewards = np.empty(N)

    for n in range(N):
        eps = 0.1 * (0.97**n)
        if n == 199:
            print("eps:", eps)
        total_reward = play_one(model, eps, gamma, lambda_)
        total_rewards[n] = total_reward

        # if (n+1) % 10 == 0:
        print("episode:", n, "total reward:", total_reward)
    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('total steps:', -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

    plot_cost_to_go(env, model)
