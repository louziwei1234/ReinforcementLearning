# 算法步骤
# 第一步，q_table给定状态值
# 第二步，计算q_value
# 第三步：选择最优策略
# 第四步：根据最优策略计算
# 第五步：更新 v

import numpy as np
import math


q_table = np.array(
    [
        [-1, -1, 0, -1, 0],
        [-1, -1, 1, 0, -1],
        [0, 1, -1, -1, 0],
        [-1, -1, -1, 0, 1],
    ]
)


def q_table_v_matrix_methods(v):
    # the state values matrix for calculating q
    # 将4x1二维数组reshape为一维数组
    v = v.reshape(-1)

    # state transfer matrix
    transfer_matrix = np.array(
        [
            [0, 1, 2, 0, 0],
            [1, 1, 3, 0, 1],
            [0, 3, 2, 2, 2],
            [1, 3, 3, 2, 3],
        ]
    )

    # the state values matrix for calculating q
    q_table_v_matrix = transfer_matrix.copy()
    for i in range(transfer_matrix.shape[0]):
        for j in range(transfer_matrix.shape[1]):
            # 向下取整再赋值
            q_table_v_matrix[i, j] = v[transfer_matrix[i, j]]

    return [q_table_v_matrix, transfer_matrix]


def value_iteration(q_tables, gamma, q_tables_v_matrix_methods):
    #  第一步，q_table给定状态值,正无穷大值（math.inf）
    n = q_tables.shape[0]
    v_new = np.random.rand(n).reshape(-1, 1)
    v = np.array([math.inf for i in range(n)]).reshape(-1, 1)

    # 迭代
    while abs(np.sum(v_new - v)) > 1e-28:
        # v存储上次迭代的S1、S2、S3、S4的state value
        v = v_new

        # 获得位置转移矩阵，和q_table计算的v矩阵
        q_table_v_matrix, transfer_matrix = q_tables_v_matrix_methods(v)

        # 第二步，计算q_value
        q_value = q_tables + gamma * q_table_v_matrix

        # 第三步：选择最优策略
        pi = np.argmax(q_value, axis=1).reshape(-1, 1)

        # 第四步：根据最优策略计算
        # 计算 R
        R = [q_tables[idx, pi[idx]] for idx in range(pi.shape[0])]

        # 计算P
        P = np.zeros((4, 4))
        for idx in range(pi.shape[0]):
            P[idx, transfer_matrix[idx, pi[idx]]] = 1

        # 第五步：更新 v
        v_new = R + (gamma * P).dot(v)

    return v_new


print('v_new update: ', value_iteration(q_table, 0.9, q_table_v_matrix_methods))

