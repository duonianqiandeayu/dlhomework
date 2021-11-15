import numpy as np

predictions = np.load("predictions.npy")

# np.savetxt('test.txt', predictions, fmt='%s', delimiter='\n')
with open('test.txt', 'w') as f:
    for path, prediction in predictions:
        print(path+' '+str(prediction))
        f.write(path+' '+str(prediction)+'\n')