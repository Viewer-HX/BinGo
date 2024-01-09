'''
    Visualize the PatchCPG instance.
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as col
import matplotlib.cm as cm

def VisualGraph(graph, state=0, options=0, show_ctrl=True, with_labels=True):

    startcolor = '#CC3366'     # R
    midcolor = '#66CC33'     # G
    endcolor = '#3366CC'   # B
    cmap2 = col.LinearSegmentedColormap.from_list('edgecol',[startcolor,midcolor,endcolor])
    # extra arguments are N=256, gamma=1.0
    cm.register_cmap(cmap=cmap2)
    csfont = {'fontname':'Times New Roman', 'size':16}
    _csfont = {'family':'Times New Roman', 'size':12}

    # get each element from graph.
    edgeIndex0 = graph['edge_index_s']
    edgeIndex1 = graph['edge_index_t']
    edgeAttr0 = graph['edge_attr_s']
    edgeAttr1 = graph['edge_attr_t']
    nodeAttr0 = graph['x_s']
    nodeAttr1 = graph['x_t']
    label = graph['y']

    # convert to numpy array.
    edgeIndex0 = edgeIndex0.numpy()
    edgeIndex1 = edgeIndex1.numpy()
    edgeAttr0 = edgeAttr0.numpy()
    edgeAttr1 = edgeAttr1.numpy()
    nodeAttr0 = nodeAttr0.numpy()
    nodeAttr1 = nodeAttr1.numpy()
    label = label.numpy()

    # construct graph.
    if (-1 == options):
        # construct graph
        G0 = nx.Graph()
        G0.add_nodes_from([n for n in range(len(nodeAttr0))])
        edgelist0 = []
        for i in reversed(range(len(edgeIndex0[0]))):
            G0.add_edge(edgeIndex0[0][i], edgeIndex0[1][i])
            edgelist0.append((edgeIndex0[0][i], edgeIndex0[1][i]))
        edgelist0 = tuple(edgelist0)
        # get node weights.
        weights0n = np.sum(np.abs(nodeAttr0), axis=1)
        weights0n = np.sqrt((weights0n - min(weights0n)) / (max(weights0n) - min(weights0n)))
        # get edge weights.
        weights0e = np.argmax(edgeAttr0, axis=1)
        dict0 = {0: 1, 1: 0.5, 2: 0}  # cfg:green, cdg:red, ddg:blue
        weights0e = tuple([dict0[w] for w in reversed(weights0e)])

        # draw figure.
        fig = plt.figure()
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G0, pos=nx.spring_layout(G0, seed=state), with_labels=with_labels,
                         font_size=8, node_color=weights0n, cmap=plt.cm.OrRd, node_size=240,
                         edgelist=edgelist0, edge_color=weights0e, width=2, edge_cmap=plt.cm.brg)
        cfg_line = mlines.Line2D([], [], color='#3366CC', label='CFG')
        cdg_line = mlines.Line2D([], [], color='#CC3366', label='CDG')
        ddg_line = mlines.Line2D([], [], color='#66CC33', label='DDG')
        plt.legend(handles=[cfg_line, cdg_line, ddg_line])
        if show_ctrl:
            plt.show()
    elif (1 == options):
        # construct graph
        G1 = nx.Graph()
        G1.add_nodes_from([n for n in range(len(nodeAttr1))])
        edgelist1 = []
        for i in reversed(range(len(edgeIndex1[0]))):
            G1.add_edge(edgeIndex1[0][i], edgeIndex1[1][i])
            edgelist1.append((edgeIndex1[0][i], edgeIndex1[1][i]))
        edgelist1 = tuple(edgelist1)
        # get node weights.
        weights1n = np.sum(np.abs(nodeAttr1), axis=1)
        weights1n = np.sqrt((weights1n - min(weights1n)) / (max(weights1n) - min(weights1n)))
        # get edge weights.
        weights1e = np.argmax(edgeAttr1, axis=1)
        dict1 = {0: 1, 1: 0.5, 2: 0}  # cfg:, cdg:, ddg:
        weights1e = tuple([dict1[w] for w in reversed(weights1e)])

        # draw figure.
        fig = plt.figure()
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G1, pos=nx.spring_layout(G1, seed=state), with_labels=with_labels,
                         font_size=8, node_color=weights1n, cmap=plt.cm.YlGn, node_size=240,
                         edgelist=edgelist1, edge_color=weights1e, width=2, edge_cmap=plt.cm.brg)
        cfg_line = mlines.Line2D([], [], color='#3366CC', label='CFG')
        cdg_line = mlines.Line2D([], [], color='#CC3366', label='CDG')
        ddg_line = mlines.Line2D([], [], color='#66CC33', label='DDG')
        plt.legend(handles=[cfg_line, cdg_line, ddg_line])
        if show_ctrl:
            plt.show()
    elif (0 == options):
        # construct graph 0
        G0 = nx.Graph()
        G0.add_nodes_from([n for n in range(len(nodeAttr0))])
        edgelist0 = []
        for i in reversed(range(len(edgeIndex0[0]))):
            G0.add_edge(edgeIndex0[0][i], edgeIndex0[1][i])
            edgelist0.append((edgeIndex0[0][i], edgeIndex0[1][i]))
        edgelist0 = tuple(edgelist0)
        # get node weights.
        weights0n = np.sum(np.abs(nodeAttr0), axis=1)
        weights0n = np.sqrt((weights0n - min(weights0n)) / (max(weights0n) - min(weights0n)))
        # get edge weights.
        weights0e = np.argmax(edgeAttr0, axis=1)
        dict0 = {0: 1, 1: 0.5, 2: 0}  # cfg:green, cdg:red, ddg:blue
        weights0e = tuple([dict0[w] for w in reversed(weights0e)])

        # construct graph 1
        G1 = nx.Graph()
        G1.add_nodes_from([n for n in range(len(nodeAttr1))])
        edgelist1 = []
        for i in reversed(range(len(edgeIndex1[0]))):
            G1.add_edge(edgeIndex1[0][i], edgeIndex1[1][i])
            edgelist1.append((edgeIndex1[0][i], edgeIndex1[1][i]))
        edgelist1 = tuple(edgelist1)
        # get node weights.
        weights1n = np.sum(np.abs(nodeAttr1), axis=1)
        weights1n = np.sqrt((weights1n - min(weights1n)) / (max(weights1n) - min(weights1n)))
        # get edge weights.
        weights1e = np.argmax(edgeAttr1, axis=1)
        dict1 = {0: 1, 1: 0.5, 2: 0}  # cfg:green, cdg:red, ddg:blue
        weights1e = tuple([dict1[w] for w in reversed(weights1e)])

        fig = plt.figure()
        # fig.subplots_adjust(wspace=0, hspace=0)
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G0, pos=nx.spring_layout(G0, seed=state), with_labels=with_labels,
                         font_size=8, node_color=weights0n, cmap=plt.cm.OrRd, node_size=240,
                         edgelist=edgelist0, edge_color=weights0e, width=2, edge_cmap=cm.get_cmap('edgecol'))
        cfg_line = mlines.Line2D([], [], color='#3366CC', label='CFG')
        cdg_line = mlines.Line2D([], [], color='#CC3366', label='CDG')
        ddg_line = mlines.Line2D([], [], color='#66CC33', label='DDG')
        plt.legend(handles=[cfg_line, cdg_line, ddg_line], prop=_csfont)
        plt.xlabel('Pre-Patch', **csfont)
        # draw G1
        plt.subplot(1, 2, 2)
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G1, pos=nx.spring_layout(G1, seed=state), with_labels=with_labels,
                         font_size=8, node_color=weights1n, cmap=plt.cm.YlGn, node_size=240,
                         edgelist=edgelist1, edge_color=weights1e, width=2, edge_cmap=cm.get_cmap('edgecol'))
        cfg_line = mlines.Line2D([], [], color='#3366CC', label='CFG')
        cdg_line = mlines.Line2D([], [], color='#CC3366', label='CDG')
        ddg_line = mlines.Line2D([], [], color='#66CC33', label='DDG')
        plt.legend(handles=[cfg_line, cdg_line, ddg_line], prop=_csfont)
        plt.xlabel('Post-Patch', **csfont)

        if show_ctrl:
            plt.show()
    else:
        print('[ERROR] <VisualGraph> argument \'options\' should be either -1 (pre), 1 (post), and 0 (both)!')
        return False

    return fig

def VisualGraphs(graph, state=0, with_labels=True):
    # get each element from graph.
    edgeIndex0 = graph['edge_index_s']
    edgeIndex1 = graph['edge_index_t']
    nodeAttr0 = graph['x_s']
    nodeAttr1 = graph['x_t']
    label = graph['y']

    # convert to numpy array.
    edgeIndex0 = edgeIndex0.numpy()
    edgeIndex1 = edgeIndex1.numpy()
    nodeAttr0 = nodeAttr0.numpy()
    nodeAttr1 = nodeAttr1.numpy()
    label = label.numpy()
    nodeNum0 = len(nodeAttr0)
    nodeNum1 = len(nodeAttr1)

    # merge two graphs.
    edgeIndex = np.c_[edgeIndex0, edgeIndex1 + nodeNum0]
    nodeAttr = np.r_[nodeAttr0, nodeAttr1]

    # construct graph.
    G = nx.Graph()
    G.add_nodes_from([n for n in range(len(nodeAttr))])
    for i in range(len(edgeIndex[0])):
        G.add_edge(edgeIndex[0][i], edgeIndex[1][i])
    # get weights.
    weights0 = np.sum(np.abs(nodeAttr0), axis=1)
    weights0 = - np.sqrt((weights0 - min(weights0)) / (max(weights0) - min(weights0)))
    weights1 = np.sum(np.abs(nodeAttr1), axis=1)
    weights1 = np.sqrt((weights1 - min(weights1)) / (max(weights1) - min(weights1)))
    weights = np.r_[weights0, weights1]
    # draw figure.
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=state),
                     with_labels=with_labels, font_size=8,
                     node_color=weights, cmap=plt.cm.RdYlGn)
    plt.show()

    return True