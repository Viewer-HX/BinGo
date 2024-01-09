from torch_geometric.data import Data

class PairData(Data):
    def __init__(self, edge_index_s, edge_attr_s, x_s, edge_index_t, edge_attr_t, x_t, y):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)