import numpy as np
import random

# 定义迷宫
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# 定义起点和终点
start = (0, 0)
goal = (4, 4)

# 动作空间：上、下、左、右
actions = ['up', 'down', 'left', 'right']

# 初始化 Q-Table，所有值为 0
# q_table 的形状将为 (5, 5, 4)，q_table[i, j, k] 表示在位置 (i, j) 处采取第 k 个动作的预期奖励。
q_table = np.zeros((maze.shape[0], maze.shape[1], len(actions)))


# 状态转换函数，确定智能体在采取某个动作后的新位置
def move(state, action):
    row, col = state
    if action == 'up' and row > 0 and maze[row-1, col] == 0:
        return (row-1, col)
    elif action == 'down' and row < maze.shape[0]-1 and maze[row+1, col] == 0:
        return (row+1, col)
    elif action == 'left' and col > 0 and maze[row, col-1] == 0:
        return (row, col-1)
    elif action == 'right' and col < maze.shape[1]-1 and maze[row, col+1] == 0:
        return (row, col+1)
    return state  # 如果移动无效，保持原位


# alpha是学习率，用于决定这次误差有多少要被学习，它本身是一个小于1的数
# gamma是对未来奖励的衰减值
# epsilon是决策上的一种策略,当epsilon=0.9时意味着90%的情况我们会按照Q表的最优解来选择行为，10%的情况随机选择行为
def q_learning(maze, start, goal, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    for _ in range(episodes):
        state = start
        while state != goal:
            # 选择动作
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)  # 探索
            else:
                action = actions[np.argmax(q_table[state])]  # 利用

            # 执行动作并观察新状态
            next_state = move(state, action)

            # 计算奖励
            reward = -1 if next_state != goal else 100  # 达到目标获得正奖励

            # 更新 Q-Table
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][actions.index(action)] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][actions.index(action)])

            # 移动到下一个状态
            state = next_state

# 运行 Q-Learning
q_learning(maze, start, goal)

# 测试策略
# 测试学到的策略，看看智能体是否能找到从起点到终点的路径
def test_policy(start, goal):
    state = start
    path = [state]
    while state != goal:
        action = actions[np.argmax(q_table[state])]
        state = move(state, action)
        path.append(state)
    return path

path = test_policy(start, goal)
print("Path found:", path)




