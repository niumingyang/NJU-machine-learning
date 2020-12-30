import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig = plt.figure(num=1, figsize=(8, 4), dpi=80)
ax = fig.add_subplot(1, 1, 1)
ax.set_title('effect of learning rate and the number of epoch on accuracy')
ax.set_xlabel('the number of epoch')
ax.set_ylabel('accuracy')
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
ax.axis([1, 10, 0.97, 1])

x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y = (0.9876, 0.9910, 0.9916, 0.9882, 0.9917, 0.9932, 0.9918, 0.9922, 0.9904, 0.9913)
ax.plot(x, y, color='blue', label='SGD learning rate=0.01 momentum=0.5')

y = (0.9847, 0.9900, 0.9915, 0.9867, 0.9923, 0.9926, 0.9930, 0.9931, 0.9929, 0.9921)
ax.plot(x, y, color='red', label='SGD learning rate=0.005 momentum=0.5')

y = (0.9709, 0.9792, 0.9846, 0.9883, 0.9859, 0.9882, 0.9906, 0.9902, 0.9907, 0.9905)
ax.plot(x, y, color='black', label='SGD learning rate=0.001 momentum=0.5')

y = (0.9747, 0.9883, 0.9895, 0.9912, 0.9920, 0.9924, 0.9925, 0.9927, 0.9922, 0.9936)
ax.plot(x, y, color='black', linestyle=':', label='SGD learning rate=0.001 momentum=0.9')

y = (0.9819, 0.9853, 0.9877, 0.9860, 0.9869, 0.9894, 0.9863, 0.9866, 0.9885, 0.9894)
ax.plot(x, y, color='yellow', linestyle=':', label='Adam with default parameters')

ax.legend()

fig2 = plt.figure(num=2, figsize=(8, 4), dpi=80)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title('learning curve of lr=0.005')
ax2.set_xlabel('the number of sammple')
ax2.set_ylabel('loss')
ax2.axis([1, 10, 0, 0.5])

x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y = (2.2389, 0.6202, 0.4395, 0.3170, 0.2285, 0.1720, 0.1345, 0.1033, 0.0837, 0.0600)
ax2.plot(x, y, color='red', label='training loss')

y = (0.0406, 0.0320, 0.0298, 0.0290, 0.0316, 0.0293, 0.0239, 0.0254, 0.0281, 0.0298)
ax2.plot(x, y, color='blue', label='validation loss')

ax2.legend()
plt.show()