import numpy as np
import sympy as sp
import random

w_11 = sp.Symbol('w_11')
w_21 = sp.Symbol('w_21')
w_12 = sp.Symbol('w_12')
w_22 = sp.Symbol('w_22')

w_1 = sp.Symbol('w_1')
w_2 = sp.Symbol('w_2')


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


X = np.array([[random.random() for j in range(2)] for i in range(100)])
# print(X)

w_y = np.array([[w_11, w_21], [w_12, w_22]])
w_z = np.array([[w_1, w_2]])

b_1 = random.random()
b_2 = random.random()

b_y = [b_1, b_2]

b_z = random.random()

Y = np.dot(X, w_y.T) + b_y

YY = np.empty((100, 2), dtype=sp.Symbol)

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        YY[i][j] = sigmoid(Y[i][j])

Z = np.dot(YY, w_z.T) + b_z

w_values = np.array([random.random() for j in range(6)])

# print(w_values)

w_dict = {w_11: w_values[0], w_21: w_values[1], w_12: w_values[2], w_22: w_values[3], w_1: w_values[4],
          w_2: w_values[5]}

grad = np.zeros((100, 6), dtype=sp.Symbol)

"Derivative with sympy."

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i][j] = sigmoid(Z[i][j])
        grad[i][j] = sp.diff(Z[i][j], w_11).subs(w_dict)
        grad[i][j + 1] = sp.diff(Z[i][j], w_21).subs(w_dict)
        grad[i][j + 2] = sp.diff(Z[i][j], w_12).subs(w_dict)
        grad[i][j + 3] = sp.diff(Z[i][j], w_22).subs(w_dict)
        grad[i][j + 4] = sp.diff(Z[i][j], w_1).subs(w_dict)
        grad[i][j + 5] = sp.diff(Z[i][j], w_2).subs(w_dict)

print(grad)

"Manual derivative."

# for i in range(grad.shape[0]):
#     grad[i, 0] = ((- Z[i][0] + Z[i][0] ** 2) * YY[i, 0] ** 2 * w_1 * np.e ** (-Y[i, 0]) * -X[i, 0]).subs(
#         w_dict)
#     grad[i, 1] = ((- Z[i][0] + Z[i][0] ** 2) * YY[i, 0] ** 2 * w_1 * np.e ** (-Y[i, 0]) * -X[i, 1]).subs(
#         w_dict)
#     grad[i, 2] = ((- Z[i][0] + Z[i][0] ** 2) * YY[i, 1] ** 2 * w_2 * np.e ** (-Y[i, 1]) * -X[i, 0]).subs(
#         w_dict)
#     grad[i, 3] = ((- Z[i][0] + Z[i][0] ** 2) * YY[i, 1] ** 2 * w_2 * np.e ** (-Y[i, 1]) * -X[i, 1]).subs(
#         w_dict)
#     grad[i, 4] = ((- Z[i][0] + Z[i][0] ** 2) * -YY[i, 0]).subs(w_dict)
#     grad[i, 5] = ((- Z[i][0] + Z[i][0] ** 2) * -YY[i, 1]).subs(w_dict)
#
# print(grad)
