import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift
from eso137 import dom, global_freqs, global_imaging_bands

xs = global_freqs * 1e-6
plt.scatter(xs, xs, label="Observation")
xs = global_imaging_bands * 1e-6
plt.scatter(
    xs, xs, label="Sky model", marker="x", s=100,
)
plt.legend()
plt.ylabel("Frequency [MHz]")
plt.xlabel("Channel")
plt.savefig("debug_channels.png")
plt.close()


N = 100
xs = np.linspace(-6, 6, N)
combs = ((1.5, 1e-3), (1, 1e-5))
for alpha, q in combs:
    op = ift.InverseGammaOperator(ift.UnstructuredDomain(N), alpha, q / dom.scalar_dvol)
    ys = op(ift.makeField(op.domain, xs)).val
    plt.plot(xs, ys, label=f"alpha {alpha}, q {q}")
plt.legend()
plt.yscale("log")
plt.savefig("prior.png")
