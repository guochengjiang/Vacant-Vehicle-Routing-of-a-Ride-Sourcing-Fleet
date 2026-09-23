"""Run regression checks without Shanghai data or a pytest dependency."""
import _bootstrap
from tests.test_notebook_parity import test_notebook_parity_and_workflows
from tests.test_small_network import test_small_network

from tests.test_legacy_small import test_legacy_small

if __name__ == '__main__':
    test_notebook_parity_and_workflows()
    test_small_network()
    test_legacy_small()
    print('PASS: original small-network BFGS and Nelder-Mead convergence and results.')
    print('PASS: grid topology, time units, policy support, independent gradient check.')
