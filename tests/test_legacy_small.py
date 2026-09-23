"""Guard the original small-network experiment against accidental replacement."""
from contextlib import redirect_stdout
import io
import numpy as np
from configs import small_network as cfg
from fleet_routing.small_experiment import make_small_network_optimizer


def test_legacy_small():
    with redirect_stdout(io.StringIO()):
        optimizer, net = make_small_network_optimizer(cfg)
        assert net['n_links'] == 76
        np.testing.assert_array_equal(optimizer.action_features[0], net['lambda_vec'])
        np.testing.assert_array_equal(optimizer.c, net['c'])
        assert optimizer.M == 1000 and optimizer.restart_interval == 10000
        bfgs = optimizer.optimize(np.array([0.0]), maxiter=100)
        nelder = optimizer.optimize_nelder_mead(np.array([0.0]), maxiter=500)
    assert bfgs.success, bfgs.message
    assert nelder.success, nelder.message
    np.testing.assert_allclose(bfgs.x, [0.04449017], atol=2e-7, rtol=0)
    np.testing.assert_allclose(nelder.x, [0.04448242], atol=2e-7, rtol=0)
    np.testing.assert_allclose([-bfgs.fun, -nelder.fun], [173.262769]*2, atol=2e-6, rtol=0)
    assert len(optimizer.iter_theta_hist) > 0
