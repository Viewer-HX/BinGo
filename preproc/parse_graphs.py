import os
import re
import sys
import time
import numpy as np
import pandas as pd
# import clang.cindex
# import clang.enumerations

# environment settings.
rootPath = './'
tempPath = './'
dataPath = rootPath + '/test_raw/'
ndatPath = tempPath + '/test_np/'
ndt2Path = tempPath + '/test_np2/'
logsPath = tempPath + '/logs/'
dictPath = tempPath + '/dict/'

# hyper-parameters.
_CodeEmbedDim_ = 32
# output parameters.
_DEBUG_  = 0
_ERROR_  = 1
# global variable.
start_time = time.time() #mark start time

# print setting.
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime

def main():
    # opCodeList = FindOpCode()
    opCodeList = np.load(dictPath + '/opcodelist.npy', allow_pickle=True)
    cnt = 0
    for root, ds, fs in os.walk(ndatPath):
        for file in fs:
            # if (cnt >= 2): continue
            # =====================================================
            if ('.DS_Store' in file):
                continue
            filename = os.path.join(root, file).replace('\\', '/')
            savename = os.path.join(ndt2Path, file)
            cnt += 1
            if os.path.exists(savename):
                print('[INFO] <main> Have the graph numpy file: [' + str(cnt) + '] ' + savename + RunTime())
                print('=====================================================')
                continue
            # =====================================================
            graph = np.load(filename, allow_pickle=True)
            edgeIndex0, edgeAttr0, nodeAttr0, edgeIndex1, edgeAttr1, nodeAttr1, label = ConstructGraph(graph, opCodeList)
            # =====================================================
            np.savez(savename, edgeIndex0=edgeIndex0, edgeAttr0=edgeAttr0, nodeAttr0=nodeAttr0, edgeIndex1=edgeIndex1, edgeAttr1=edgeAttr1, nodeAttr1=nodeAttr1, label=label)
            print('[INFO] <main> save the graph information into numpy file: [' + str(cnt) + '] ' + savename + RunTime())
            print('=====================================================')
            # =====================================================

    return

def FindOpCode():
    '''
    :return:
    '''

    opCodeList = []
    cnt = 0
    for root, ds, fs in os.walk(ndatPath):
        for file in fs:
            cnt += 1
            # if (cnt >= 2): continue
            # =====================================================
            filename = os.path.join(root, file).replace('\\', '/')
            graph = np.load(filename, allow_pickle=True)
            # =====================================================
            nodeAttr = graph['nodeAttr0']
            # print('===================')
            # print(filename)
            # print(nodeAttr)
            opCode = [op for node in nodeAttr for op in node]
            # print(opCode)
            opCode = {}.fromkeys(opCode)
            opCode = list(opCode.keys())
            # print(opCode)
            opCodeList.extend(opCode)
            # =====================================================
            nodeAttr = graph['nodeAttr1']
            # print(nodeAttr)
            opCode = [op for node in nodeAttr for op in node]
            # print(opCode)
            opCode = {}.fromkeys(opCode)
            opCode = list(opCode.keys())
            # print(opCode)
            opCodeList.extend(opCode)
            # =====================================================
            opCodeList = {}.fromkeys(opCodeList)
            opCodeList = list(opCodeList.keys())
            # print(opCodeList)
            # =====================================================

    np.save(dictPath + '/opcodelist.npy', opCodeList, allow_pickle=True)
    print('[INFO] <main> save the op code list into numpy file [' + str(len(opCodeList)) + '] : ' + dictPath + '/opcodelist.npy' + RunTime())
    # print('[INFO] <main> save the op code dictionary into numpy file: ' + dictPath + '/opcodedict.np/' + RunTime())

    return opCodeList

def ConstructGraph(graph, opCodeList):
    edgeIndex0 = graph['edgeIndex0']
    edgeIndex1 = graph['edgeIndex1']
    edgeAttr0 = graph['edgeAttr0']
    edgeAttr1 = graph['edgeAttr1']

    nodeAttr0 = []
    for node in graph['nodeAttr0']:
        attr = [list(node).count(opCode) for opCode in opCodeList]
        nodeAttr0.append(attr)
    # print(nodeAttr0)

    nodeAttr1 = []
    for node in graph['nodeAttr1']:
        attr = [list(node).count(opCode) for opCode in opCodeList]
        nodeAttr1.append(attr)
    # print(nodeAttr1)

    label = graph['label']

    return edgeIndex0, edgeAttr0, nodeAttr0, edgeIndex1, edgeAttr1, nodeAttr1, label

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'construct_graphs.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # check folders.
    if not os.path.exists(dictPath):
        os.makedirs(dictPath)
    if not os.path.exists(ndt2Path):
        os.makedirs(ndt2Path)
    # main entrance.
    main()