import numpy as np
import os
from perceptron import read_double_matrix

print(np.random.random((3, 3)))

a = np.matrix([1, 2, 3, 4])
print(a)
b = np.matrix('1; -2; 3; 4')
print(b)
f = np.array([1, 2, 2, 4])
print(f)

print(np.sum(b.T == f))
# print(a*b)

print(os.getcwd())
open('orinput.txt')

# print(a * 5)
# print(np.heaviside(b, 1))

# c = np.matrix([0.5])
# print('hans')
# print(np.heaviside(c, 1))

# c = read_double_matrix("xorinput.txt")

# d = np.random.uniform(-0.1, 0.1, (1,4))
e = np.random.uniform(-0.1, 0.1, (4,1))
# f = np.matrix(e)
# print(d)
print(e)
# print(d * e)
# print(d * f)
# print(os.getcwd())
## print(c)