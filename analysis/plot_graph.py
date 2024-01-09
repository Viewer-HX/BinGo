'''
    Visualize the Patch-CPG.
'''

import os
import numpy as np
import torch
from libs.pairdata import PairData
from libs.graphvis import VisualGraph
import argparse

np.set_printoptions(threshold=np.inf)

rootPath = './'
tempPath = './temp/'
mdatPath = rootPath + '/data_np/'
ndatPath = rootPath + '/data_np2/'
ndatFile = ''
saveFile = ''

_SHOWMAP = True
_SHOWFIG = False
_OPTION = 0
_STATE = 0

def main():
    if os.path.exists(ndatFile):
        ndatFilePath = ndatFile
    elif ndatFile in os.listdir(ndatPath):
        ndatFilePath = os.path.join(ndatPath, ndatFile).replace('\\', '/')
    else:
        print('[ERROR] Cannot find the file ', ndatFile)
        return

    if _SHOWMAP:
        FindCodeMapping(ndatFilePath)

    print('[INFO] Read graph file from ', ndatFilePath)
    graph = ReadGraph(ndatFilePath)
    graphfig = VisualGraph(graph, state=_STATE, options=_OPTION, show_ctrl=_SHOWFIG)

    # save figure.
    if (0 == len(saveFile)):
        pathseg = ndatFilePath.split(sep='/')
        verinfo = 'pre' if (-1 == _OPTION) else 'post' if (1 == _OPTION) else 'all'
        filename = pathseg[-1][:-4] + '_' + verinfo + '_st' + str(_STATE) + '.png'
        # filename = pathseg[-1][:-4] + '_' + verinfo + '_st' + str(_STATE) + '.pdf'
        filename = os.path.join(tempPath, filename).replace('\\', '/')
        if not os.path.exists(tempPath):
            os.mkdir(tempPath)
    else:
        filename = saveFile

    print('[INFO] Save the figure in ', filename)
    graphfig.savefig(filename, dpi=graphfig.dpi, bbox_inches='tight', pad_inches=0.02)
    # graphfig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.02)

    return

def FindCodeMapping(fp):
    fp_split = fp.split('/')
    filename = fp_split[-1]
    if filename in os.listdir(mdatPath):
        filepath = os.path.join(mdatPath, filename).replace('\\', '/')
    else:
        print('[WARNING] Cannot find the file ', filename)
        return

    g = np.load(filepath, allow_pickle=True)
    nodeAttr0 = g['nodeAttr0']
    nodeDict0 = g['nodeDict0'].item()
    mapDict0 = {node: id for node, id in enumerate(nodeDict0)}
    nodeAttr1 = g['nodeAttr1']
    nodeDict1 = g['nodeDict1'].item()
    mapDict1 = {node: id for node, id in enumerate(nodeDict1)}

    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    pSave = os.path.join(tempPath, fp_split[-1][:-4] + '.txt')
    fSave = open(pSave, 'w')

    # pre
    print('Pre-Patch:')
    fSave.write('Pre-Patch:\n')
    for i in range(len(mapDict0)):
        print(i, '\t', mapDict0[i], '\t', end='')
        fSave.write(str(i) + ',' + mapDict0[i] + ',')
        for j in range(len(nodeAttr0[i])):
            print(nodeAttr0[i][j], ' ', end='')
            fSave.write(nodeAttr0[i][j] + ',')
        print()
        fSave.write('\n')
    # post
    print('Post-Patch:')
    fSave.write('Post-Patch:\n')
    for i in range(len(mapDict1)):
        print(i, '\t', mapDict1[i], '\t', end='')
        fSave.write(str(i) + ',' + mapDict1[i] + ',')
        for j in range(len(nodeAttr1[i])):
            print(nodeAttr1[i][j], ' ', end='')
            fSave.write(nodeAttr1[i][j] + ',')
        print()
        fSave.write('\n')

    fSave.close()
    print('[INFO] Save the mapping dictionary to ', pSave)

    return True

def ReadGraph(filename):

    graph = np.load(os.path.join(filename), allow_pickle=True)
    # sparse each element.
    edgeIndex0 = torch.tensor(graph['edgeIndex0'], dtype=torch.long)
    nodeAttr0 = torch.tensor(graph['nodeAttr0'], dtype=torch.float)
    edgeAttr0 = torch.tensor(graph['edgeAttr0'], dtype=torch.float)
    edgeIndex1 = torch.tensor(graph['edgeIndex1'], dtype=torch.long)
    nodeAttr1 = torch.tensor(graph['nodeAttr1'], dtype=torch.float)
    edgeAttr1 = torch.tensor(graph['edgeAttr1'], dtype=torch.float)
    label = torch.tensor(graph['label'], dtype=torch.long)
    # construct an instance of torch_geometric.data.Data.
    data = PairData(edge_index_s=edgeIndex0, edge_attr_s=edgeAttr0, x_s=nodeAttr0,
                    edge_index_t=edgeIndex1, edge_attr_t=edgeAttr1, x_t=nodeAttr1, y=label)

    return data

def ArgsParser():
    # define argument parser.
    parser = argparse.ArgumentParser()
    # add arguments.
    parser.add_argument('-npz', help='the graph npz file name.', required=True)
    parser.add_argument('-save', help='the graph save file path.')
    parser.add_argument('-option', help='the version option of graph. (-1:pre; 1:post; 0:both)', type=int)
    parser.add_argument('-state', help='the init state of graph.', type=int)
    parser.add_argument('-show', help='show the figure', action='store_true')
    # parse the arguments.
    args = parser.parse_args()
    # global variables.
    global ndatFile
    global _SHOWFIG
    global saveFile
    global _STATE
    global _OPTION
    # perform actions.
    if (args.npz): ndatFile = args.npz
    if (args.show): _SHOWFIG = True
    if (args.save): saveFile = args.save
    if (args.state): _STATE = args.state
    if (args.option): _OPTION = args.option

    return parser

if __name__ == '__main__':
    # sparse the arguments.
    ArgsParser()
    # main
    main()