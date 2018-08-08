'''
plot history.yml
'''

import sys, yaml
with open(sys.argv[1],'r') as f:
    history = yaml.load(f)

import matplotlib.pyplot as plt
print(history['test_loss'], history['test_acc'])

plt.clf()
plt.plot(history['loss'], 'b', label='train loss')
plt.plot(history['val_loss'], 'r', label='valid loss')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(fontsize=10)
plt.draw()
plt.show()

plt.clf()
plt.plot(history['acc'], 'b', label='train accuracy')
plt.plot(history['val_acc'], 'r', label='valid accuracy')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend(fontsize=10)
plt.draw()
plt.show()
