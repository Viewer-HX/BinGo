'''
    extract graph
'''

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
logsPath = tempPath + '/logs/'

# hyper-parameters.
# _CodeEmbedDim_ = 32
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
            # if (cnt >= 1): continue
            # =====================================================
            if ('.DS_Store' in file):
                continue
            filename = os.path.join(root, file).replace('\\', '/')
            savename = os.path.join(ndatPath, file + '.npz')
            cnt += 1
            # if os.path.exists(savename):
            #     print('[INFO] <main> Have the graph numpy file: [' + str(cnt) + '] ' + savename + RunTime())
            #     print('=====================================================')
            #     continue
            # =====================================================
            nodes, edges = ReadFile(filename)
            nodeDict0, edgeIndex0, edgeAttr0 = ProcEdges(edges, version='-1')
            nodeAttr0 = ProcNodes(nodes, nodeDict0, version='-1')
            nodeDict1, edgeIndex1, edgeAttr1 = ProcEdges(edges, version='1')
            nodeAttr1 = ProcNodes(nodes, nodeDict1, version='1')
            label = [1] if ('positives' in root) else [0]
            if (0 == len(edgeAttr0)) or (0 == len(edgeAttr1)):
                print('[WARNING] <main> A graph is empty! Sample is dropped.')
                print('=====================================================')
                continue
            # =====================================================
            np.savez(savename, edgeIndex0=edgeIndex0, edgeAttr0=edgeAttr0, nodeAttr0=nodeAttr0, nodeDict0=nodeDict0,
                     edgeIndex1=edgeIndex1, edgeAttr1=edgeAttr1, nodeAttr1=nodeAttr1, nodeDict1=nodeDict1, label=label)
            print('[INFO] <main> save the graph information into numpy file: [' + str(cnt) + '] ' + savename + RunTime())
            print('=====================================================')
            # =====================================================
    return

def ReadFile(filename):
    '''
    :param filename:
    :return: nodesData ['34', 'mov', .., '1']  edgesData [['0','1','CFG','-1'], ]
    '''

    # read lines from the file.
    print('[INFO] <ReadFile> Read data from:', filename)
    fp = open(filename, encoding='utf-8', errors='ignore')
    lines = fp.readlines()
    fp.close()
    if _DEBUG_: print(lines)

    # get the data from edge and node information.
    signEdge = 1
    edgesData = []
    nodesData = []
    for line in lines:
        # for each line in this file.
        if line.startswith('==='):
            signEdge = 0
        elif (1 == signEdge):
            # Edge:
            contents = re.findall(r'\(\d+,\d+,[CFGD]+,-?\d\)', line)
            if 0 == len(contents):
                if _ERROR_: print('[ERROR] <ReadFile> Edge does not match the format, para:', filename, line)
                continue
            content = contents[0]       # get the first match.
            content = content[1:-1].replace(' ', '')    # remove () and SPACE.
            segs = content.split(',')   # split with comma.
            edgesData.append(segs)
        elif (0 == signEdge):
            # Node:
            contents = re.findall(r'\(\d+,.*,-?\d\)', line)
            if 0 == len(contents):
                if _ERROR_: print('[ERROR] <ReadFile> Node does not match the format, para:', filename, line)
                continue
            content = contents[0]  # get the first match.
            content = content[1:-1]  # remove ().
            segs = content.split(',')   # split with comma.
            nodesData.append(segs)
        else:
            # Error:
            if _ERROR_: print('[ERROR] <ReadFile> Neither an edge or a node, para:', filename, line)

    print('[INFO] <ReadFile> Read', len(nodesData), 'nodes and', len(edgesData), 'edges.' + RunTime())
    if _DEBUG_:
        print(nodesData)
        print(edgesData)

    return nodesData, edgesData

def ProcEdges(edgesData, version='-1'):
    '''
    Mapping the edges to edge embeddings.
    '''

    edges = [edge for edge in edgesData if edge[3] == version]
    if (0 == len(edges)):
        print('[WARNING] <ProcEdges> [Version ' + version + '] Find 0 edge in the sub-graph.' + RunTime())
        return {}, [], []
    # get the node set.
    nodesout = [edge[0] for edge in edges]
    nodesin = [edge[1] for edge in edges]
    nodeset = nodesout + nodesin
    # remove duplicates
    nodeset = {}.fromkeys(nodeset)
    nodeset = list(nodeset.keys())
    # get the dictionary.
    nodeDict = {node: index for index, node in enumerate(nodeset)}
    print('[INFO] <ProcEdges> [Version ' + version + '] Find', len(nodeDict), 'nodes connected with', len(edges), 'edges.' + RunTime())
    if _DEBUG_: print(nodeDict)

    # get the edge index. [2 * edge_num]
    nodesoutIndex = [nodeDict[node] for node in nodesout]
    nodesinIndex = [nodeDict[node] for node in nodesin]
    edgeIndex = np.array([nodesoutIndex, nodesinIndex])
    print('[INFO] <ProcEdges> [Version ' + version + '] Get', len(edgeIndex), '*', len(edgeIndex[0]), 'edge index array.' + RunTime())
    if _DEBUG_: print(edgeIndex)

    # get the dictionary of type.
    typeDict = {'CFG': [1, 0, 0], 'CDG': [0, 1, 0], 'DDG': [0, 0, 1]}

    # get the edge attributes. [edge_num, num_edge_features]
    edgeAttr = np.array([typeDict[edge[2]] for edge in edgesData if edge[3] == version])
    print('[INFO] <ProcEdges> [Version ' + version + '] Get', len(edgeAttr), '*', len(edgeAttr[0]), 'edge attribute array.' + RunTime())
    if _DEBUG_: print(edgeAttr)

    return nodeDict, edgeIndex, edgeAttr

def ProcNodes(nodesData, nodeDict, version='0'):
    '''
    Mapping the nodes to node embeddings.
    '''

    nodes = [node for node in nodesData if node[-1] == version]
    if (0 == len(nodes)):
        print('[WARNING] <ProcNodes> [Version ' + version + '] Find 0 node in the sub-graph.')
        return []
    nodeList = [node[0] for node in nodes]
    if _DEBUG_: print(nodesData)
    if _DEBUG_: print(nodes)
    if _DEBUG_: print(nodeList)
    if _DEBUG_: print(nodeDict)

    # check the integrity of the node list.
    for node in nodeDict:
        if node not in nodeList:
            print('[ERROR] <ProcNodes> [Version ' + version + '] Node', node, 'does not in node list.')
            return []

    # get the node attributes with the order of node dictionary.
    nodeOrder = [nodeList.index(node) for node in nodeDict]
    nodesDataNew = [nodes[order] for order in nodeOrder]

    nodeAttr = []
    for nodeData in nodesDataNew:
        # for each node.
        attrList = nodeData[1:-1]
        # append the node attribute.
        nodeAttr.append(attrList)

    nodeAttr = np.array(nodeAttr, dtype=object)
    if _DEBUG_: print(nodeAttr)
    print('[INFO] <ProcNodes> [Version ' + version + '] Get', len(nodeAttr), '* n node attribute array.' + RunTime())

    return nodeAttr

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'extract_graphs.txt'
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