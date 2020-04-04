# %%
from dask.distributed import Client, progress
import time
import numpy as np
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler


def foo(N):
    # print(f"Start:{N}")
    for _ in range(10):
        N = np.sqrt(N + N)
        x = np.random.rand(1000, 1000)
        x = x / 5

    # time.sleep(1)
    # print(f"End:{N}")
    return {"dumb": N, "dumber": N * N}


if __name__ == "__main__":
    client = Client()  # set up local cluster on your laptop
    with Profiler() as prof, ResourceProfiler(
        dt=0.25
    ) as rprof, CacheProfiler() as cprof:
        A = client.map(foo, np.arange(0.1, 10, 0.1))

    # progress(A)

    # prof.visualize()
    x = client.gather(A)
    print(x)


# %%
