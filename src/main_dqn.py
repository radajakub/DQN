import numpy as np
import matplotlib.pyplot as plt
from DQN import DeepQLearning

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

            action = dqn.choose_action(state)

            next_state, reward, terminal = env.step(action)

            return_value = reward + (gamma * return_value)
            step_count += 1
            dqn.store_transition(state, action, reward, next_state)

            if (step > 200) and (step % 5 == 0):
                dqn.learn()
            #### Begin learning after accumulating certain amount of memory #####
            state = next_state

            if terminal:

                print(" {} End. Total steps : {}\n".format(
                    episode + 1, step_count))
                break

            step += 1

        # add return value of the current episode to the list
        returns.append(return_value)

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
    plt.title(f'DQN on {envtype}: Returns per episode')
    plt.xlabel('episodes')
    plt.ylabel('return')
    plt.plot(results, c='b', label='returns')
    plt.plot(running_means, c='r', label='100 window running mean')
    plt.legend(loc='lower right')
    plt.savefig(f'dqn_{envtype}_returns.png')
    # ==================================

    print('Game over.\n')
    env.destroy()


if __name__ == "__main__":

    env = Robot_Gridworld()

    dqn = DeepQLearning(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.9,
                        e_greedy=0.05,
                        replace_target_iter=50,
                        memory_size=3000,
                        batch_size=32)

    env.after(100, update)  # Basic module in tkinter
    env.mainloop()  # Basic module in tkinter
