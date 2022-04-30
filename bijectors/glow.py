import normflow as nf

class SimpleGlow():
    def __init__(self, K, n_channels, hidden_channels):
        super().__init__()
        self.bijectors = [nf.flows.GlowBlock(n_channels, hidden_channels, split_mode='checkerboard') for _ in range(K)]

    def forward(self, x):
        y, log_j_final = self.bijectors[-1].forward(x)
        for i in reversed(range(len(self.bijectors)-1)):
            y, log_j = self.bijectors[i].forward(y)
            log_j_final += log_j
        return y, log_j_final


    def inverse(self, y):
        x, log_j_final = self.bijectors[0].inverse(y)
        for i in range(1, len(self.bijectors)):
            x, log_j = self.bijectors[i].inverse(x)
            log_j_final += log_j
        return x, log_j_final

