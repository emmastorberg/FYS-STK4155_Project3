import numpy as np
import pytest
import utils
from Diffusion1D import *

@pytest.mark.parametrize("dx", [0.11, 0.09, 0.3, 0.6, 0.21])
def test_dx_values(dx):
    with pytest.raises(ValueError):
        sol = Diffusion1D(dx=dx)

@pytest.mark.parametrize("dt", [0.1, 0.01, 0.008])
def test_dt_values(dt):
    with pytest.raises(ValueError):
        sol = Diffusion1D(dx=0.1, dt = dt)

@pytest.mark.parametrize("dx", [0.1, 0.01])
def test_diffusion_equation_error(dx):
    Nt = 100

    sol = Diffusion1D(dx=dx)
    data = sol(Nt, save_step=1)

    data = utils.dict_to_matrix(data)

    x = sol.x
    t = np.linspace(0, Nt*sol.dt, Nt + 1)

    u = utils.analytical_solution(x, t)

    if u.shape != data.shape:
        raise ValueError("The numerical solution and analytical solution do not have the same shape.")
    
    assert np.allclose(data, u, atol=1e-02)

if __name__ == "__main__":
    test_diffusion_equation_error()