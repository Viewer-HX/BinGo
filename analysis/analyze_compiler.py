import os
import numpy as np

def func(lst):
    ret = [[0, 0], [0, 0]]
    for l in lst:
        ret[l[2]][l[3]] += 1
    print(ret, int((ret[0][0]+ret[0][1]+ret[1][0]+ret[1][1])))

    precision = ret[1][1] / (ret[1][0] + ret[1][1]) if (ret[1][0] + ret[1][1]) else 0
    recall = ret[1][1] / (ret[0][1] + ret[1][1]) if (ret[0][1] + ret[1][1]) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    acc = (ret[0][0] + ret[1][1]) / (ret[0][0] + ret[1][1] + ret[0][1] + ret[1][0])
    fp = ret[1][0]  / (ret[1][0] + ret[0][0]) if (ret[1][0] + ret[0][0]) else 0
    fn = ret[0][1]  / (ret[0][1] + ret[1][1]) if (ret[0][1] + ret[1][1]) else 0
    print(f'acc: {acc}, F1: {F1}, FN:{fn}, FP {fp}')

    return ret

posList = os.listdir('./data_raw/positives/')
negList = os.listdir('./data_raw/negatives/')
lines = open('./logs/TestResults.txt').readlines()

results = []
for line in lines:
    res = []

    [filename, label] = line.split(',')
    filename = filename[12:-4]
    label = int(label[0])

    info = filename.split('-')
    res.append(info[1])
    res.append(info[2])
    res.append(label)

    if filename in posList:
        res.append(1)
    elif filename in negList:
        res.append(0)
    else:
        print('ERROR')
        pass

    results.append(res)


func(results)
for compl in ['gcc', 'clang']:
    for optlv in ['no', 'O0', 'O1', 'O2', 'O3', 'Os']:
        r = []
        for res in results:
            if res[0] == compl and res[1] == optlv:
                r.append(res)

        print(compl, optlv)
        func(r)


for compl in ['gcc', 'clang']:
    r = []
    for res in results:
        if res[0] == compl:
            r.append(res)

    print(compl)
    func(r)



