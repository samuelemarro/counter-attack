import torch
inp = torch.randn(1, 3, 10, 12)
w = torch.randn(2, 3, 4, 5)

# Handmade conv
inp_unf = torch.nn.functional.unfold(inp, (4, 5))
print(inp_unf.shape)
print(w.view(w.size(0), -1).t().shape)
out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
print(out_unf.shape)
out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
print(out.shape)
# or equivalently (and avoiding a copy),
# out = out_unf.view(1, 2, 7, 8)

# Check that the result is correct
print((torch.nn.functional.conv2d(inp, w) - out).abs().max())