import torch
import sys


N=2000
dim=40
device = 'cuda'

m = torch.randn((dim, dim)).to(device)
print(f"rank of the matrix: {torch.linalg.matrix_rank(m)}")
FIM = 0
GFIM = 0

I_r = torch.eye(dim).to(device)

for i  in range(N):
    m = torch.randn((dim, dim)).to(device)
    m_flatten = m.reshape(-1, 1)

    GFIM += torch.kron(I_r, m @ m.T / dim)
    FIM +=  m_flatten @ m_flatten.T

GFIM = GFIM/N
FIM = FIM/N
    





min_GFIM = torch.min(GFIM).item()
max_GFIM = torch.max(GFIM).item()


min_FIM = torch.min(FIM).item()
max_FIM = torch.max(FIM).item()

min_value = min(min_GFIM, min_FIM)
max_value = max(max_GFIM, max_FIM)
print(f"min_value: {min_value}, max_value: {max_value}")
## plot the GFIM and FIM
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(GFIM.cpu().numpy())
plt.clim(min_value, max_value)
plt.colorbar()
plt.title("GFIM")

plt.subplot(1, 2, 2)
plt.imshow(FIM.cpu().numpy())
plt.clim(min_value, max_value)
plt.colorbar()
plt.title("FIM")

plt.tight_layout()
plt.savefig("GFIM_FIM.pdf", bbox_inches='tight')
