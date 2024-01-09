import os
import numpy as np

posList = os.listdir('./data_raw/positives/')
negList = os.listdir('./data_raw/negatives/')
patchList = os.listdir('./data_np2/')

npres = []
for f in patchList:
    # print(f)
    graph = np.load('./data_np2/'+f, allow_pickle=True)
    # print(list(graph.keys()))
    label = graph['label'][0]
    nodes0 = len(graph['nodeAttr0'])
    nodes1 = len(graph['nodeAttr1'])
    edges0 = len(graph['edgeAttr0'])
    edges1 = len(graph['edgeAttr1'])
    info = f[12:-4].split('-')
    compl = 0 if info[1] == 'gcc' else 1
    opts = {'no': -1, 'O0': 0, 'O1': 1, 'O2': 2, 'O3': 3, 'Os': 4}
    optlv = opts[info[2]]
    npres.append([nodes0, nodes1, edges0, edges1, label, compl, optlv])
# npres = np.array(npres)
# print(npres)

for l in [0, 1]:
    for compl in [0, 1]:
        for optlv in [0, 1, 2, 3, 4]:
            nums = []
            # print(l, 'gcc' if compl == 0 else 'clang', optlv)
            for n in npres:
                if l == n[-3] and compl == n[-2] and optlv == n[-1]:
                    nums.append([n[0], n[1], n[2], n[3], n[1]/n[0], n[3]/n[2]])

            # print(nums)
            nums = np.array(nums)
            nums = sum(nums) / len(nums)
            print(nums[0], nums[1], nums[2], nums[3])
            # print()






