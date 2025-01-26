# import time
# import jax
# import jax.numpy as jnp

# def test_jax_gpu():
#     print("Available devices:", jax.devices())

#     # Create two big matrices on the GPU
#     x = jnp.ones((4000, 4000))
#     y = jnp.ones((4000, 4000))

#     # Time a single matrix multiplication
#     t0 = time.time()
#     z = jnp.dot(x, y)
#     # Force the computation to finish on the GPU
#     _ = z.block_until_ready()
#     t1 = time.time()

#     print(f"Time to do a 4000x4000 matrix multiply: {t1 - t0:.4f} seconds")

# if __name__ == "__main__":
#     test_jax_gpu()

# import jax
# import jax.numpy as jnp
# import numpy as np

# def main():
#     rng = jax.random.PRNGKey(42)
#     # Simple split
#     rng1, rng2 = jax.random.split(rng)
#     print("Split worked:", rng1, rng2)

#     print(jax.random.randint(rng, [], minval=0, maxval=np.iinfo(np.int32).max))

# if __name__ == "__main__":
#     main()


# import nvidia.cudnn

# # version = nvidia.cudnn.getVersion()
# print("cuDNN version:", nvidia.cudnn.__file__)


import jax
from jax.lib import xla_bridge

backend = xla_bridge.get_backend()
print("Default backend:", jax.default_backend())
print("Platform version:", backend.platform_version)

