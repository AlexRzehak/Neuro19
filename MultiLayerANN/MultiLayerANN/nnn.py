import numpy as np

# def wurst(prev_layer_size, layer_size):
#     return np.random.rand(prev_layer_size+1, layer_size)

# w = wurst(2, 4)

# print(w)

# liste = [wurst(2, 4), wurst(1, 2)]

# print(liste)

# w3 = liste[0][0:-1]

# # w2 = w[0:-1]

# print(w3)

# print(3 * w3)

a = np.array([1, 2, 3, 5])
b = np.array([1, 2, 3, 5])

c = np.outer(a, b)

print(4 * c)


# def hans(baum: str, *urgs):
#     print(baum)
#     print(urgs)
#     print(len(urgs))

# print(wurst(2,4))

# hans('kanns', 5, 3, 7)

# a = [6,5]
# a[0] = a[0] -1
# print(a[0])
# print(a[1])

# y = np.array([1, 2, 3, 4])
# print(y)
# def mugo(array):
#     f = lambda x, y: x + y
#     return f(array, array)


# print(mugo(y))

# class Sigmoid:
#     @staticmethod
#     def f(x: np.array) -> np.array:
#         """ the sigmoid function """
#         # TODO
#         sick = lambda x: 1/(1 + np.exp(-x))
#         return sick(x)
    
#     @staticmethod
#     def d(x: np.array) -> np.array:
#         """ the first derivative """
#         # TODO
#         f = Sigmoid.f(x)
#         der = lambda x: x * (1 - x)
#         return der(f)

# gg = np.array([-1, -0.5, 0, 0.5, 1])

# dd = [1, 2, 3, 5]
# bb = np.multiply(dd, dd)

# print(bb)

# print(Sigmoid.f(gg))
# print(Sigmoid.d(gg))
# print(np.tanh(gg))