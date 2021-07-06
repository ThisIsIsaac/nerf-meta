import torch
import torch.nn.functional as F
class ExampleModel(torch.nn.Module):

    def __init__(self, dim) -> None:
        super(ExampleModel, self).__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.linear_gen = torch.nn.Linear(dim, dim)

    def forward(self, x, y, mode):
        res = self.linear_gen(y)


        out1 = self.linear(x)
        if mode == 'original':
            out2 = self.linear(out1)
        elif mode == 'detach':
            out2 = F.linear(out1, self.linear.weight.detach(),
                            self.linear.bias.detach())
        elif mode == 'req_grad':
            self.linear.requires_grad_(False)
            out2 = self.linear(out1)
            # self.linear.requires_grad_(True)
        else:
            raise RuntimeError
        return out2


torch.manual_seed(2809)
# Random input output data for this example
N, D = 64, 100
x = torch.randn(N, D)
y = torch.randn(N, D)

model = ExampleModel(D)
criterion = torch.nn.MSELoss(reduction='sum')
weight_ref = model.linear.weight.clone()


# Detach
y_pred = model(x, mode='detach')
loss = criterion(y_pred, y)
loss.backward()

weight_detach = model.linear.weight.clone()
model.zero_grad()

# Req_grad
y_pred = model(x, mode='req_grad')
loss = criterion(y_pred, y)
loss.backward()

weight_req_grad = model.linear.weight.clone()
model.zero_grad()

# Compare
print((weight_ref - weight_detach).abs().max())
print((weight_req_grad - weight_detach).abs().max())