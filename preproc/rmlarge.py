# remove large graphs

import os
import numpy as np

dataPath = './data_np2/'
NODE_THR = 100

cnt_d = 0
cnt_r = 0
for root, _, ds in os.walk(dataPath):
    for file in ds:
        if '.DS_Store' in file: continue
        filename = os.path.join(root, file)
        graph = np.load(filename, allow_pickle=True)
        numNode0 = len(graph['nodeAttr0'])
        numNode1 = len(graph['nodeAttr1'])
        if (numNode0 > NODE_THR) or (numNode1 > NODE_THR):
            os.remove(filename)
            print(f'[INFO] Removing {filename} ({numNode0}, {numNode1})')
            cnt_d += 1
        else:
            cnt_r += 1
print(f'[INFO] Totally removed {cnt_d} graphs. ({cnt_r} remaining)')