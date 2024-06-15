# %%
"""
https://qiita.com/pocokhc/items/f7ab56051bb936740b8f
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as tD
import torchvision.transforms as tT
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class EDLayer(nn.Module):
    def __init__(
        self,
        input_num: int,
        neuron_num: int,
        is_positive: bool = True,
        bias: bool = True,
    ) -> None:
        super(EDLayer, self).__init__()
        self.is_positive = is_positive

        self.pos_weights = nn.Parameter(torch.rand(input_num, neuron_num))
        self.neg_weights = nn.Parameter(torch.rand(input_num, neuron_num))
        if bias:
            self.pos_bias = nn.Parameter(torch.rand(1, neuron_num))
            self.neg_bias = nn.Parameter(torch.rand(1, neuron_num))
        else:
            self.pos_bias = None
            self.neg_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_inputs = x
        neg_inputs = x
        if self.pos_bias is not None:
            return self.sign * (
                (pos_inputs @ self.pos_weights + self.pos_bias) 
                + (neg_inputs @ self.neg_weights + self.neg_bias)
            )
        else:
            return self.sign * (
                pos_inputs @ self.pos_weights 
                + neg_inputs @ self.neg_weights
            )
    
    @property
    def sign(self):
        return 1 if self.is_positive else -1



batch_size = 64
input_num = 3 * 32 * 32
neurons_num = 1024
output_dim = 10

train_dataset = tD.CIFAR10(root="./data", train=True, download=True, transform=tT.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = tD.CIFAR10(root="./data", train=False, download=True, transform=tT.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

neuron = nn.Sequential(
    EDLayer(input_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, neurons_num),
    nn.ReLU(),
    EDLayer(neurons_num, output_dim),
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"classifier has {count_parameters(neuron)} parameters")

neuron.train()
neuron.cuda()
optimizer = torch.optim.Adam(neuron.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    count = 0
    true_count = 0
    for x, t in pbar:
        x = x.view(x.size(0), -1).cuda()
        t = t.cuda()
        y = neuron(x)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += x.size(0)
        true_count += (y.argmax(dim=1) == t).sum().item()
        pbar.set_description(f"epoch:{epoch:02d} loss: {loss.item():.4f} acc: {true_count / count:.4f}")


neuron.eval()

test_true_count = 0
test_count = 0
with torch.inference_mode():
    for x, t in test_dataloader:
        x = x.view(x.size(0), -1).cuda()
        t = t.cuda()
        y = neuron(x)
        test_count += x.size(0)
        test_true_count += (y.argmax(dim=1) == t).sum().item()

print(x.shape, y.shape)
print(f"test acc: {test_true_count / test_count:.4f}")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
x, t = next(iter(test_dataloader))
with torch.inference_mode():
    x = x.view(x.size(0), -1).cuda()
    y = neuron(x)

print(x.shape, y.shape)
k = 4
x, t = x[:k ** 2], t[:k ** 2]

fig, axes = plt.subplots(k, k, figsize=(12, 12))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x[i].cpu().view(3, 32, 32).permute(1, 2, 0).numpy())
    pred_i = y[i].argmax().item()
    ax.set_title(f"pred: {train_dataset.classes[pred_i]} true: {train_dataset.classes[t[i].item()]}")
    ax.axis("off")
plt.show()

# %%

print(neuron.state_dict())