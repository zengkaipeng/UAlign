import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.MultiheadAttention(64, 4, batch_first=True)
        self.pred = torch.nn.Linear(64, 1)

    def forward(self, x, src_mask=None):
        attn_o, attn_w = self.model(x, x, x, key_padding_mask=src_mask)
        if src_mask is not None:
            attn_o = attn_o[~src_mask]
        return self.pred(attn_o).squeeze(dim=-1)

x = torch.randn(4, 5, 64)
y = torch.zeros(4, 5).bool()
y[1] = True

label = torch.randn(4, 5)
label[label > 0] = 1
label[label <= 0] = 0

model = Model()

result = model(x, y)

print(result)
print(label[~y])

loss = torch.nn.functional.binary_cross_entropy_with_logits(
	result, label[~y]
)

loss.backward()
for k, v in model.named_parameters():
	print(k, v.grad)
