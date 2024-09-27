import torch
import torch.nn as nn

class GenerateNN(nn.Module):
    def __init__(self, out_dims) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        self.stem = self.make_layer(in_dim=2, out_dim=out_dims[0])
        for i in range(1, len(out_dims)):
            self.layers.append(self.make_layer(in_dim=out_dims[i-1], out_dim=out_dims[i]))
        
        self.final = nn.Linear(out_dims[-1], out_features=2)

        self.t_emb = nn.Embedding(num_embeddings=(1000), embedding_dim=out_dims[0])

    def make_layer(self, in_dim, out_dim):

        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, t):

        t_emb = self.t_emb(t)

        x = self.stem(x)
        x = x + t_emb

        for _, layer in enumerate(self.layers):
            x = layer(x)

        out = self.final(x)

        return out


if __name__=="__main__":
    
    model = GenerateNN(out_dims=[64, 128, 256, 512])

    x = torch.randn((4, 100, 2))
    t = torch.randint(1, 1000, (4, 1))

    print(model(x, t).shape)