class GCN_2layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, skip = False):
        super(GCN_2layer, self).__init__()
        self.skip = skip
        
        self.gcl1 = GraphConv(in_features, hidden_features)
        
        if self.skip:
            self.gcl_skip = GraphConv(hidden_features, out_features, activation = 'softmax', skip = self.skip,
                                  skip_in_features = in_features)
        else:
            self.gcl2 = GraphConv(hidden_features, out_features, activation = 'softmax')
        
    def forward(self, A, X):
        out = self.gcl1(A, X)
        if self.skip:
            out = self.gcl_skip(A, out, X)
        else:
            out = self.gcl2(A, out)
            
        return out