import numpy as np
a = np.loadtxt('treestrap_weights.csv', delimiter = ',')
absolute_list = np.abs(a)
for i in range(len(a)):
    if absolute_list[i] < 3:
        a[i] = 0
print(a)

np.savetxt("pruned_weights.csv", a, delimiter=",")
