# 算法步骤
# 第一步，给定策略
# 第二步，计算状态值
# 第三步：使用第二节状态值迭代法获得该策略下收敛的状态值
# 第四步：由该状态值更新最优策略

import math

import numpy as np

q_table_r = np.array(
   [
       [-1, -1, 0, -1, 0],
       [-1, -1, 1, 0, -1],
       [0, 1, -1, -1, 0],
       [-1, -1, -1, 0, 1],
   ]
)

# state transfer matrix
q_table_transfer_matrix = np.array(
   [
       [0, 1, 2, 0, 0],
       [1, 1, 3, 0, 1],
       [0, 3, 2, 2, 2],
       [1, 3, 3, 2, 3],
   ]
)


def q_table_v_matrix_methods(v, transfer_matrix):
   # the state values matrix for calculating q
   v = v.reshape(-1)

   # the state values matrix for calculating q
   q_table_v_matrix = transfer_matrix.copy()
   for i in range(transfer_matrix.shape[0]):
       for j in range(transfer_matrix.shape[1]):
           q_table_v_matrix[i, j] = v[transfer_matrix[i, j]]

   return q_table_v_matrix


## 严格按照定义方式
def iterative_solution(R, P, gamma):
   # n_iter 为迭代次数
   # 初始化  vπ
   n = R.shape[0]
   v_new = np.random.rand(n, 1).reshape(-1, 1)
   v = np.array([math.inf for i in range(n)]).reshape(-1, 1)

   while abs(np.sum(v_new - v)) > 1e-8:
       v = v_new
       v_new = R + (gamma * P).dot(v)
   return v_new


def policy_iteration(q_table, transfer_matrix, gamma, q_table_v_matrix_methods):
   # 第一步，随机给定策略
   action_n = q_table.shape[1]
   state_n = q_table.shape[0]
   # 给定策略
   pi = np.random.randint(low=0, high=action_n, size=state_n)

   # 当状态值收敛时停止
   v = np.array([0 for i in range(state_n)]).reshape(-1, 1)
   v_new = np.array([math.inf for i in range(state_n)]).reshape(-1, 1)
   while abs(np.sum(v_new - v)) > 1e-8:
       v = v_new
       # 第二步，计算状态值
       # 计算 R
       R = np.array([q_table[idx, pi[idx]] for idx in range(pi.shape[0])])

       # 计算P
       P = np.zeros((state_n, state_n))
       for idx in range(pi.shape[0]):
           P[idx, transfer_matrix[idx, pi[idx]]] = 1
       # 计算该策略下的状态值
       v_new = iterative_solution(R, P, gamma)

       ## 第三步,由该状态值更新最优策略
       # 计算 q_table
       q_table_v_matrix = q_table_v_matrix_methods(v_new, transfer_matrix)

       # 计算q_value
       q_value = q_table + gamma * q_table_v_matrix

       # 选择最优策略
       pi = np.argmax(q_value, axis=1).reshape(-1, 1)

   return pi


print('pi update: ', policy_iteration(q_table_r, q_table_transfer_matrix, 0.9, q_table_v_matrix_methods))

