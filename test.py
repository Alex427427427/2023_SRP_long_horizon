import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])
d = np.array([10, 11, 12])
e = np.array([13,14,15, 16])
list = [a, b, c, d, e]
print(list)

s1 = np.copy(a)
for i in range(1, len(list)):
    s1 = np.vstack((s1, list[i]))

print(s1)
