import numpy as np
import matplotlib.pyplot as plt
from Reinforce import Reinforce

# from env1 import Robot_Gridworld
# envtype = 'env1'

from env2 import Robot_Gridworld
envtype = 'env2'

gamma = 0.99
returns = []


def update():
    step = 0

    for episode in range(1000):

        state = env.reset()

        step_count = 0
        return_value = 0

        while True:

            env.render()

            action = Reinforce.choose_action(state)
            next_state, reward, terminal = env.step(action)

            return_value = reward + (gamma * return_value)
            step_count += 1
            # dqn.store_transition(state, action, reward, next_state)
            Reinforce.rewards.append(reward)
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state

            if terminal:

                print(" {} End. Total steps : {}\n".format(
                    episode + 1, step_count))
                break

            if step_count > 1000:
                break

            step += 1

        returns.append(return_value)
        Reinforce.learn()

    # ==== Plot returns per episode ====
    # I will also do a running mean with window of 100 episodes to smooth the plot
    # in other case I would do an average over multiple runs but since we have a single
    #  run here, this seems as a good choice
    results = np.array(returns)
    half_window = 50
    running_means = np.array([np.average(results[i-half_window:i+half_window])
                             for i in range(half_window, results.shape[0]-half_window)])
    running_means = np.concatenate([np.full(
        half_window, running_means[0]), running_means, np.full(half_window, running_means[-1])])
    plt.figure(1)
    plt.title(f'REINFORCE on {envtype}: Returns per episode')
    plt.xlabel('episodes')
    plt.ylabel('return')
    plt.plot(results, c='b', label='returns')
    plt.plot(running_means, c='r', label='100 window running mean')
    plt.legend(loc='lower right')
    plt.savefig(f'rl_{envtype}_returns.png')
    # ==================================

    print('Game over.\n')
    env.destroy()


if __name__ == "__main__":

    env = Robot_Gridworld()

    Reinforce = Reinforce(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          discount_factor=0.9,
                          eps=0.1,)

    env.after(100, update)  # Basic module in tkinter
    env.mainloop()  # Basic module in tkinter
