import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("means", "rb") as means_file:
	means = pickle.load(means_file)

with open("stds", "rb") as stds_file:
	stds = pickle.load(stds_file)


with open("means_Georgia", "rb") as means_file:
	means_Georgia = pickle.load(means_file)

with open("stds_Georgia", "rb") as stds_file:
	stds_Georgia = pickle.load(stds_file)

means_Georgia = means_Georgia[0].reshape(1, means_Georgia.shape[1], 1)
stds_Georgia = stds_Georgia[0].reshape(1, stds_Georgia.shape[1], 1)

print(means.shape)
print(stds.shape)
print(means_Georgia.shape)
print(stds_Georgia.shape)

plt.hist(means[0, :, 0], bins=50)
plt.hist(means_Georgia[0, :, 0], bins=50)
plt.gca().set(title='Georgia (orange) vs PTBXL (blu), lead D1', ylabel='Frequency');
plt.savefig("distr.png")