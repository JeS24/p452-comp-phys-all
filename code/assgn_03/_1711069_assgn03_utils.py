import numpy as np
import itertools

from plotly import graph_objects as go
from scipy.optimize import newton


# Q1 - MC Integration
def mlcg(seed, a=1664525, m=2**32):
    """
    Multiplicative / Lehmer Linear Congruential Generator
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    seed: float
        Seed
    a: float
        Multiplier
        Defaults to ``1664525``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    float:
        Sequence of Pseudo-Random Numbers, having a period, that is sensitive to the choice of a and m.

    """
    while True:
        seed = (a * seed) % m
        yield seed

def mlcgList(N:int, range:tuple, seed:float=42, a:float=1664525, m:float=2**32):
    """
    Continuous
    Returns normalized list of MLCG-generated random values
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    N: int
        Number of random numbers to be generated
    range: tuple
        Range from where the numbers will be sampled
    seed: float
        Seed
        Defaults to ``42``
    a: float
        Multiplier
        Defaults to ``1664525``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    numpy.ndarray
        Normalized list of MLCG-generated random values

    """    
    start, end = range

    rnList = np.array(list(itertools.islice(mlcg(seed, a, m), 0, N))) / m
    return end * rnList + start


# Q2 - Inifinite Potential Well
def rk4_solver(f, x0, t, V, E):
    NT = len(t)
    x = np.array([x0] * NT)

    for i in np.arange(NT - 1):
        dt = t[i + 1] - t[i]

        k1 = dt * f(x[i], x[i], V[i], E)
        k2 = dt * f(x[i] + 0.5 * k1, x[i] + 0.5 * dt, V[i], E)
        k3 = dt * f(x[i] + 0.5 * k2, x[i] + 0.5 * dt, V[i], E)
        k4 = dt * f(x[i] + k3, x[i + 1], V[i], E)

        x[i + 1] = x[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x

def norm(psi):
    """
    Normalizes input wavefunction
    
    """
    return psi / np.max(psi)

def shoot_multi(f, psi_init, x, V, E_guesses):
    """
    Returns all wavefunction values at the right boundary for each energy guess

    """
    psi_b = []

    for E in E_guesses:
        psi = rk4_solver(f, psi_init, x, V, E)
        psi_b.append(psi[-1, 0])

    return np.asarray(psi_b)

def get_E(f, psi_init, x, V, E_guesses):
    """
    Returns energy eigenvalues
    Note that these E values also make the ODE consistent with the boundary values and hence solvable.

    """
    def shoot_once(E, f, psi0, x, V):
        """
        f(E), used to find zeros using Newton-Raphson method

        """
        psi = rk4_solver(f, psi0, x, V, E)

        return psi[-1, 0]

    # Obtaining an array of wavefunction values at the right boundary considering the energy_guesses
    psi_b_arr = shoot_multi(f, psi_init, x, V, E_guesses)
    # Checking for where the sign of the point-wise derivative at the right boundary changes
    approx_zeros = np.where(np.diff(np.signbit(psi_b_arr)))[0]
    # For each such sign-change, using Newton-Raphson method to find the nearby zeros
    E_list = []
    for _0 in approx_zeros:
        E_list.append(newton(shoot_once, E_guesses[_0], args=(f, psi_init, x, V)))

    return np.asarray(E_list)

def schröd_eq(y, r, V, E):
    """
    1D Schrödinger Equation for Infinite Potential Well
    
    """
    psi, dpsi = y
    ddpsi = [dpsi, (V - E) * psi]

    return np.asarray(ddpsi)

def inf_pot_well(psi_init, dx):
    """
    Sets up the system, gets energy eigenvalues and solves the system

    """
    # Setting up the system
    x_arr = np.arange(0.0, 1.0 + dx, dx)
    V = np.zeros(len(x_arr))
    E_guesses = np.arange(1.0, 100.0, 5.0)

    # Getting energy eigenvalues based on multiple solutions using the shooting method
    E_eig = get_E(schröd_eq, psi_init, x_arr, V, E_guesses)

    # Final solution with E_eig
    psi = []
    for eig in E_eig:
        res = rk4_solver(schröd_eq, psi_init, x_arr, V, eig)
        psi.append(norm(res[:, 0]))

    psi = np.asarray(psi)

    return E_eig, x_arr, psi

def inf_pot_well_th(dx):
    """
    Analytical solution for Infinite Potential Well

    """
    x_arr = np.arange(0.0, 1.0 + dx, dx)
    kappa = np.arange(1.0, 4.0, 1.0)

    sol = []
    for k in kappa:
        sol.append(np.sin(k * np.pi * x_arr))

    return np.asarray(sol)

def plot_wf(title:str, x, psi, psi_th, spacing:int=5):
    """
    Plots the calculated and theoretical wavefunction

    """
    # Plotting the calculated curve
    # Spaced 5 points apart for better visibility with respect to the theoretical curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[::spacing],
        y=psi[::spacing],
        mode='markers',
        marker_symbol='diamond',
        name="Calculated"
    ))
    # Plotting the theoretical curve
    fig.add_trace(go.Scatter(
        x=x,
        y=norm(psi_th),
        mode='lines',
        name="Analytical"
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        title_x=0.5,
        title_y=0.9,
        width=800,
        height=600,
        xaxis_title=r'$\hat{x}$',
        yaxis_title=r'$\Psi(\hat{x})$',
        paper_bgcolor='#FFF',
        plot_bgcolor='#FFF'
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='black')

    return fig
