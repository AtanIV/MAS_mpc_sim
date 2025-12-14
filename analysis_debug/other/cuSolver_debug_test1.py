import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

# ----------------------------
# 1. Trivial cuSolver test
# ----------------------------
print("=== Test 1: trivial linear solve ===")
try:
    A = jnp.array([[1.0]])
    b = jnp.array([1.0])
    x = jnp.linalg.solve(A, b)
    print("Trivial solve succeeded:", x)
except Exception as e:
    print("Trivial solve failed with:", e)


# ----------------------------
# 2. Tiny GraphsTuple mock
# ----------------------------
class GraphsTuple(NamedTuple):
    nodes: jnp.ndarray
    edges: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray
    states: jnp.ndarray


# Fake normalizer just passes through
def fake_add_edge_feats(graph, states):
    return graph


# Mock evaluator (scalar h + jacobian vector)
@jax.jit
def evaluate_h_and_jacobian_jax(graph: GraphsTuple) -> Tuple[jax.Array, jax.Array]:
    # pretend "h" is sum of node positions
    h_ego = jnp.sum(graph.nodes)

    # pretend jacobian is gradient of sum wrt nodes
    def cbf(nodes):
        return jnp.sum(nodes)

    jacobian = jax.grad(cbf)(graph.nodes)  # should be all ones
    return h_ego, jacobian


print("\n=== Test 2: toy graph evaluation ===")
try:
    toy_graph = GraphsTuple(
        nodes=jnp.array([1.0, 2.0, 3.0]),
        edges=jnp.array([0.0]),       # not used
        senders=jnp.array([0]),       # not used
        receivers=jnp.array([0]),     # not used
        states=jnp.array([0.0])       # not used
    )
    h, jac = evaluate_h_and_jacobian_jax(toy_graph)
    print("Graph eval succeeded: h =", h, ", jac =", jac)
except Exception as e:
    print("Graph eval failed with:", e)


# /home/tung656/anaconda3/envs/gcbfplus/bin/python /home/tung656/MastersFiles/Y2/MAS_Simulations_mpc/cuSolver_debug_test1.py
# === Test 1: trivial linear solve ===
# E0930 15:22:11.430333   18632 pjrt_stream_executor_client.cc:2916] Execution of replica 0 failed: INTERNAL: jaxlib/gpu/solver_handle_pool.cc:37: operation gpusolverDnCreate(&handle) failed: cuSolver internal error
# Trivial solve failed with: INTERNAL: jaxlib/gpu/solver_handle_pool.cc:37: operation gpusolverDnCreate(&handle) failed: cuSolver internal error
#
# === Test 2: toy graph evaluation ===
# Graph eval succeeded: h = 6.0 , jac = [1. 1. 1.]
#
# Process finished with exit code 0

# Perfect test — this result tells us a lot:
#
#     Test 1 (trivial jnp.linalg.solve) failed → that means your environment has the cuSolver / JAX / CUDA wheel mismatch issue we talked about. Even the simplest GPU linear algebra call is broken.
#
#     Test 2 (toy graph eval) succeeded → that’s only because the mock version never called any GPU solver routines. Your real cbf_jacobian_jax does trigger them via autodiff, so it blows up inside SQP.
#
# So the root cause is definitely the JAX/CUDA installation, not your SQP code.
