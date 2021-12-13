import matplotlib.pyplot as plt
import numpy as np

values = []
with open('log_file.txt', 'r') as f:
    line = f.readline()
    while line: 
        seg = line.split(';')[-2]
        values.append(float(seg[seg.find(':')+1:]))
        line = f.readline()

plt.figure()
plt.plot(np.arange(len(values))*1000, values)
plt.xlabel('Frames')
plt.ylabel('Average Score')
plt.savefig('result.png')

