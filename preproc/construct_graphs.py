'''
    extract graph
'''

import os
import sys
import time
import numpy as np
import pandas as pd

# environment settings.
rootPath = './'
tempPath = './'
dataPath = rootPath + '/data_npz/'
ndatPath = tempPath + '/data_np2/'
logsPath = tempPath + '/logs/'

# hyper-parameters.
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
    cnt = 0
    for root, ds, fs in os.walk(dataPath):
        for file in fs:
            # if (cnt >= 10): continue
            # =====================================================
            if ('.DS_Store' in file):
                continue
            filename = os.path.join(root, file).replace('\\', '/')
            savename = os.path.join(ndatPath, file)
            cnt += 1
            # if os.path.exists(savename):
            #     print('[INFO] <main> Have the graph numpy file: [' + str(cnt) + '] ' + savename + RunTime())
            #     print('=====================================================')
            #     continue
            # =====================================================
            graph = np.load(filename, allow_pickle=True)
            nodes = graph['nodesData']
            edges = graph['edgesData']
            label = graph['label']
            nodeDict0, edgeIndex0, edgeAttr0 = ProcEdges(edges, version=-1)
            nodeAttr0 = ProcNodes(nodes, nodeDict0, version=-1)
            nodeDict1, edgeIndex1, edgeAttr1 = ProcEdges(edges, version=1)
            nodeAttr1 = ProcNodes(nodes, nodeDict1, version=1)
            if (0 == len(nodeDict0)) or (0 == len(nodeDict1)):
                print('[WARNING] <main> A graph is empty! Sample is dropped.')
                # print('=====================================================')
                # continue
            # =====================================================
            np.savez(savename, edgeIndex0=edgeIndex0, edgeAttr0=edgeAttr0, nodeAttr0=nodeAttr0, nodeDict0=nodeDict0,
                     edgeIndex1=edgeIndex1, edgeAttr1=edgeAttr1, nodeAttr1=nodeAttr1, nodeDict1=nodeDict1, label=label)
            print('[INFO] <main> save the graph information into numpy file: [' + str(cnt) + '] ' + savename + RunTime())
            print('=====================================================')
            # =====================================================
    return

def ProcEdges(edgesData, version=-1):
    '''
    Mapping the edges to edge embeddings.
    '''

    edges = [edge for edge in edgesData if edge[3] == version]
    if (0 == len(edges)):
        print(f'[WARNING] <ProcEdges> [Version {version}] Find 0 edge in the sub-graph.' + RunTime())
        return {}, np.array([[0],[1]]), np.zeros((1,3))
    # get the node set.
    nodesout = [edge[0] for edge in edges]
    nodesin = [edge[1] for edge in edges]
    nodeset = nodesout + nodesin

    # remove duplicates
    nodeset = {}.fromkeys(nodeset)
    nodeset = list(nodeset.keys())
    # get the dictionary.
    nodeDict = {node: index for index, node in enumerate(nodeset)}
    print(f'[INFO] <ProcEdges> [Version {version}] Find {len(nodeDict)} nodes connected with {len(edges)} edges.' + RunTime())
    if _DEBUG_: print(nodeDict)

    # get the edge index. [2 * edge_num]
    nodesoutIndex = [nodeDict[node] for node in nodesout]
    nodesinIndex = [nodeDict[node] for node in nodesin]
    edgeIndex = np.array([nodesoutIndex, nodesinIndex])
    print(f'[INFO] <ProcEdges> [Version {version}] Get {len(edgeIndex)} * {len(edgeIndex[0])} edge index array.' + RunTime())
    if _DEBUG_: print(edgeIndex)

    # get the dictionary of type.
    typeDict = {'CFG': [1, 0, 0], 'CDG': [0, 1, 0], 'DDG': [0, 0, 1]}

    # get the edge attributes. [edge_num, num_edge_features]
    edgeAttr = np.array([typeDict[edge[2]] for edge in edgesData if edge[3] == version])
    print(f'[INFO] <ProcEdges> [Version {version}] Get {len(edgeAttr)} * {len(edgeAttr[0])} edge attribute array.' + RunTime())
    if _DEBUG_: print(edgeAttr)

    return nodeDict, edgeIndex, edgeAttr

def ProcNodes(nodesData, nodeDict, version=0):
    '''
    Mapping the nodes to node embeddings.
    '''

    nodes = [node for node in nodesData if node[1] == version]
    if (0 == len(nodes)) or (0 == len(nodeDict)):
        print(f'[WARNING] <ProcNodes> [Version {version}] Find 0 node in the sub-graph.')
        return np.zeros((2, len(nodesData[0,-1])))
    nodeList = [node[0] for node in nodes]
    if _DEBUG_: print(nodesData)
    if _DEBUG_: print(nodes)
    if _DEBUG_: print(nodeList)
    if _DEBUG_: print(nodeDict)

    # check the integrity of the node list.
    for node in nodeDict:
        if node not in nodeList:
            print(f'[ERROR] <ProcNodes> [Version {version}] Node {node} does not in node list.')
            return []

    # get the node attributes with the order of node dictionary.
    nodeOrder = [nodeList.index(node) for node in nodeDict]
    nodesDataNew = [nodes[order] for order in nodeOrder]

    nodeAttr = []
    for nodeData in nodesDataNew:
        # for each node.
        attrList = nodeData[-1]
        # append the node attribute.
        nodeAttr.append(attrList)

    nodeAttr = np.array(nodeAttr)
    if _DEBUG_: print(nodeAttr)
    print(f'[INFO] <ProcNodes> [Version {version}] Get {len(nodeAttr)} * n node attribute array.' + RunTime())

    return nodeAttr

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'construct_graphs.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # check folders.
    if not os.path.exists(ndatPath):
        os.makedirs(ndatPath)
    # main entrance.
    main()