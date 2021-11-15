
import numpy as np
import matplotlib.pyplot as plt
import torchvision

history = np.load("./save/history5.npy")
print(history)

plt.plot(history[:,0],history[:, 1:3])
plt.legend(['train loss', 'valid loss'])
plt.xlabel('epoch num')
plt.ylabel('loss')
plt.ylim(0, 5)
plt.grid()
plt.savefig('./save/ex2fig/loss_curve5.png')
plt.show()

plt.cla()
plt.plot(history[:,0],history[:, 3:5])
plt.legend(['train acc', 'valid acc'])
plt.xlabel('epoch num')
plt.ylabel('acc')
plt.ylim(0,1)
plt.grid()
plt.savefig('./save/ex2fig/acc_curve5.png')
plt.show()