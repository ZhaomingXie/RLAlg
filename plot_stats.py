import pickle
import numpy as np
import matplotlib.pyplot as plt
import statistics

class Stats:
	def __init__(self, file):
		with open (file, 'rb') as fp:
			stats = pickle.load(fp)
		self.mean = stats[2]
		self.noisy_mean = stats[4]
		self.std = stats[3]
		self.samples = stats[1]
		self.low = []
		self.high = []
		for i in range(len(self.mean)):
			self.low.append(self.mean[i]-self.std[i])
			self.high.append(self.mean[i]+self.std[i])
			#self.samples.append(i)
	def plot(self, ax, color='g', variance_color='lightgreen', label=''):
		ax.plot(self.noisy_mean, color, label=label)
		#ax.fill_between(self.samples, self.low, self.high, alpha=1, color=variance_color)


stats_walker_no_contact_1 = Stats("stats/walker2d_no_contact_seed8_Iter201.stat")
stats_walker_contact_1 = Stats("stats/walker2d_contact_seed8_Iter201.stat")
stats_walker_contact_2 = Stats("stats/walker2d_contact_seed16_Iter201.stat")

fig = plt.figure()
ax = fig.add_subplot(111)

stats_walker_no_contact_1.plot(ax, color='r', label='Walker_Non_Partition', variance_color='salmon')

stats_walker_contact_1.plot(ax, color='g', label='Walker_Partition', variance_color='lightgreen')
stats_walker_contact_2.plot(ax, color='g', label='Walker_Partition', variance_color='lightgreen')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_xlim([0,201])
plt.legend(loc='upper left')
plt.show()