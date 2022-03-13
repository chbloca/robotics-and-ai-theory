import numpy as np
import random
import gym
import time

env = gym.make("Taxi-v2")
next_reward = np.zeros((500,6 ))


#Training
total_episodes = 50000
total_test_episodes = 100
max_steps = 99
learning_rate = 0.7
gamma = 0.618
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

Total_reward = []
Total_actions = []
for i in range(10):
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0,1)

            if exp_exp_tradeoff > epsilon:
                action = np.argmax(next_reward[state,:])

            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            next_reward[state, action] = next_reward[state, action] + learning_rate * (reward + gamma * np.max(next_reward[new_state, :]) - next_reward[state, action])

            state = new_state

            if done == True:
                break

        episode += 1

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

    # Testing
    test_tot_reward = 0
    test_tot_actions = 0
    past_observation = -1
    observation = env.reset();
    for t in range(50):
        test_tot_actions = test_tot_actions+1
        action = np.argmax(next_reward[observation])
        if (observation == past_observation):
            action = random.sample(range(0,6),1)
            action = action[0]
        past_observation = observation
        observation, reward, done, info = env.step(action)
        test_tot_reward = test_tot_reward+reward
        #env.render()
        time.sleep(1)
        if done:
            break
    print("Total reward: ")
    print(test_tot_reward)
    print("Total actions: ")
    print(test_tot_actions)

    Total_reward.append(test_tot_reward)
    Total_actions.append(test_tot_actions)

Total_reward_avg = np.average(Total_reward)
print("Total_reward_avg: "Total_reward_avg)
Total_actions_avg = np.average(Total_actions)
print(Total_actions_avg)

