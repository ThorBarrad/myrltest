import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# R
R = 2

delta_t = 0.15

# V
V = 4.2
W = V / R * delta_t

# v
v = 1

sheep_actions = np.linspace(0, np.pi, 6)

# start place (r, \theta)
dog_start = [R, 0]
sheep_start = [0, 0]

replay_buffer = []
max_replay_size = 300
replay_get = 20

max_term = 3000

epsilon = 0.1
gamma = 0.9


def dog_move(state):
    dog_r, dog_theta, sheep_r, sheep_theta = state
    dog_theta = dog_theta % (2 * np.pi)
    sheep_theta = sheep_theta % (2 * np.pi)
    if (sheep_theta - dog_theta) % np.pi == 0:
        dog_theta += np.random.choice([-1, 1]) * W
    elif (sheep_theta - dog_theta) % (2 * np.pi) < np.pi:
        dog_theta += W
    else:
        dog_theta -= W
    return dog_r, dog_theta, sheep_r, sheep_theta


def sheep_move(state, action):
    dog_r, dog_theta, sheep_r, sheep_theta = state
    sheep_x, sheep_y = sheep_r * np.cos(sheep_theta), sheep_r * np.sin(sheep_theta)
    theta = np.pi / 2 - sheep_theta + action
    sheep_x -= v * delta_t * np.cos(theta)
    sheep_y += v * delta_t * np.sin(theta)
    sheep_r = np.sqrt(sheep_x ** 2 + sheep_y ** 2)
    if sheep_x > 0:
        sheep_theta = np.arctan(sheep_y / sheep_x)
    else:
        sheep_theta = np.arctan(sheep_y / sheep_x) + np.pi
    return dog_r, dog_theta, sheep_r, sheep_theta


def is_end(state):
    dog_r, dog_theta, sheep_r, sheep_theta = state
    return sheep_r >= R


def is_win(state):
    dog_r, dog_theta, sheep_r, sheep_theta = state
    return sheep_r > R and ((dog_theta - sheep_theta) % (2 * np.pi)) > W


# create Q network
Q_model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
                              tf.keras.layers.Dense(1)])
Q_model.compile(optimizer='adam', loss='mse')
'''
a = np.zeros([6, 5])
b = np.ones([6, 1])
Q_model.fit(a, b)
'''
# initial Q_network
# Q_model.fit(np.random.random([10, 5]), np.random.random([10, 1]), epochs=50)


def epsilon_greedy_action(state, action):
    if np.random.random() <= epsilon:
        return np.random.choice(action)
    else:
        action_value = [Q_model.predict(np.array([[state[0], state[1], state[2], state[3], action_]]))[0][0] for action_ in action]
        next_choice = []
        max_value = max(action_value)
        for i in range(len(action_value)):
            if action_value[i] == max_value:
                next_choice.append(action[i])
        return np.random.choice(next_choice)


ep = 0
update_network = 0
while ep < max_term:
    ep += 1
    cur_state = [R, 0, 0, 0]

    plt.ion()
    fig = plt.figure(1)
    ax = plt.subplot(111, projection='polar')

    while not is_end(cur_state):
        update_network += 1

        cur_state = dog_move(cur_state)

        cur_action = epsilon_greedy_action(cur_state, sheep_actions)
        next_state = sheep_move(cur_state, cur_action)

        reward = min((next_state[1] - next_state[3]) % (2 * np.pi), (- next_state[1] + next_state[3]) % (2 * np.pi)) * next_state[2]

        if len(replay_buffer) < max_replay_size:
            replay_buffer.append([cur_state, cur_action, reward, next_state, is_end(next_state)])
        else:
            replay_buffer = replay_buffer[1:]
            replay_buffer.append([cur_state, cur_action, reward, next_state, is_end(next_state)])

        cur_state = next_state

        if update_network % 10 == 1:
            mini_replay_buffer = []

            for i in range(replay_get):
                mini_replay_buffer.append(replay_buffer[np.random.randint(len(replay_buffer))])

            train_y = []
            train_x = []
            for each_replay in mini_replay_buffer:
                cur_state_j, cur_action_j, reward_j, next_state_j, is_end_j = each_replay[0], each_replay[1], each_replay[2], each_replay[3], each_replay[4]
                if is_end_j:
                    y_j = reward_j
                else:
                    y_j = reward_j + gamma * max([Q_model.predict(np.array([[next_state_j[0], next_state_j[1], next_state_j[2], next_state_j[3], action_]]))[0][0] for action_ in sheep_actions])
                train_y.append([y_j])
                train_x.append([cur_state_j[0], cur_state_j[1], cur_state_j[2], cur_state_j[3], cur_action_j])
            train_X = np.array(train_x)
            train_Y = np.array(train_y)
            Q_model.fit(train_X, train_Y, epochs=100)

        theta = [cur_state[1], cur_state[3]]
        r = [cur_state[0], cur_state[2]]
        ax.scatter(theta, r)
        plt.pause(0.005)
        plt.show()
    plt.clf()
    plt.ioff()

