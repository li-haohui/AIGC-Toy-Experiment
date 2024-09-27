from numpy import dtype
import torch
import torch.nn as nn
import torch.distributions as D
from matplotlib import pyplot as plt

from torch.utils.data import Dataset

class MixedGaussianDistribution(Dataset):
    def __init__(self, means: list, covars: list, weights: list, num_samples: int =10) -> None:
        super().__init__()

        weights = [w/sum(weights) for w in weights]

        self.gaussians = []
        
        for i in range(len(means)):
            self.gaussians.append({
                "mean": means[i], "covar": covars[i], "weight": weights[i]
            })

        self.num_gaussians = len(self.gaussians)
        self.num_samples = num_samples

        print(self.gaussians[0]["covar"].shape)
    
    def __len__(self):
        return 1000

    def __getitem__(self, index):

        sample = torch.zeros((self.num_samples, 2), dtype=torch.float32)

        for gaussian in self.gaussians:
            sampler = D.MultivariateNormal(loc=gaussian["mean"], covariance_matrix=gaussian["covar"])

            sample += gaussian["weight"] * sampler.sample((self.num_samples, ))

        return sample


if __name__=="__main__":

    means = [torch.randint(1, 10, size=(2,), dtype=torch.float) for _ in range(8)]

    covars = []
    for _ in range(8):
        A = torch.rand((2, 2))
        covars.append(A @ A.T)

    weights = [i for i in range(1, 8+1)]

    print(covars)

    GMM = MixedGaussianDistribution(
        means = means,
        covars=covars,
        weights=weights,
        num_samples=100
    )

    data = next(iter(GMM)).numpy()

    
    plt.scatter([d[0] for d in data], [d[1] for d in data])
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig("./data.png")



    