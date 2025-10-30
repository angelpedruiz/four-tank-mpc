import numpy as np

class FourTankStochastic:
    """
    Stochastic model of the Modified Four Tank System. 
    The disturbances are piecewise constant stochastic.
    There is also measurement noise.

    States: mass of liquid in tanks [m1, m2, m3, m4] [g]
    Manipulated variables: pump flows [F1, F2] [cm^3/s]
    Disturbance variables: unmeasured flow rates [F3, F4] [cm^3/s]
    Parameters: pipe areas, tank areas, flow splits, gravity, density
    """

    def __init__(self, params: dict, x0=None, dt: float=0.01, measurement_noise_std: float=0.0):
        """
        Initialize the four-tank system.

        Parameters
        ----------
        params : ndarray, shape (12,)
            System parameters:
                params[0:4]   -> pipe cross-sectional areas a [cm^2]
                params[4:8]   -> tank cross-sectional areas A [cm^2]
                params[8:10]  -> flow distribution ratios gamma [-]
                params[10]    -> gravity g [cm/s^2]
                params[11]    -> density rho [g/cm^3]
        x0 : ndarray, shape (4,), optional
            Initial states (mass in each tank). Defaults to zeros.
        dt : float, optional
            Time step for the simulation. Default is 0.01 s.
        measurement_noise_std : float
            Standard deviation of the measurement noise.
        """
        self.params = params
        self.x0 = x0 if x0 is not None else np.zeros(4)
        self.dt = dt
        self.measurement_noise_std = measurement_noise_std

    def dynamics(self, x, u, d=None):
        """
        Deterministic dynamics in the form: dx = f(x(t),u(t),d(t),p) dt
        
        Compute the derivatives of the system states for piecewise constant disturbances.

        Parameters
        ----------
        t : float
            Current time (required by ODE solvers)
        x : ndarray, shape (4,)
            Current states (mass in tanks) 
        u : ndarray, shape (2,)
            Manipulated variables (pump flows)
        d: ndarray, shape (2,), optional
            Piecewise constant disturbance variables d(t) = dk for tk≤t<tk+1. Defaults to zeros.

        Returns
        -------
        xdot : ndarray, shape (4,)
            Time derivatives ẋ(t) = f(x(t),u(t),d(t),p)
        """
        a = self.params[:4]
        A = self.params[4:8]
        gamma = self.params[8:10]
        g = self.params[10]
        rho = self.params[11]
        
        F1 = u[0]
        F2 = u[1]
        F3 = d[0]
        F4 = d[1]
        
        # Inflows
        qin = np.zeros(4)
        qin[0] = gamma[0] * F1
        qin[1] = gamma[1] * F2
        qin[2] = (1 - gamma[1]) * F2
        qin[3] = (1 - gamma[0]) * F1

        # Heights
        h = x / (rho * A)

        # Outflows
        qout = a * np.sqrt(2 * g * h)

        # Mass balances
        xdot = np.zeros(4)
        xdot[0] = rho * (qin[0] + qout[2] - qout[0])
        xdot[1] = rho * (qin[1] + qout[3] - qout[1])
        xdot[2] = rho * (qin[2] - qout[2] + F3)
        xdot[3] = rho * (qin[3] - qout[3] + F4)

        return xdot
    
    def measurement(self, x):
        """
        Measurement model: y(t) = g(x(t),p) + v(t) with v(t) ~ N(0,Rvv(p))
        
        Compute the measurements from the states with Gaussian measurement noise.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Current states (mass in tanks)

        Returns
        -------
        y : ndarray, shape (4,)
            Noisy measurements y(t) = g(x(t),p) + v(t) where g maps states to heights
        """
        A = self.params[4:8]
        rho = self.params[11]
        
        # g(x(t),p): deterministic measurement function (mass to height)
        h = x / (rho * A)
        
        # v(t) ~ N(0, Rvv(p)): measurement noise
        Rvv = (self.measurement_noise_std ** 2) * np.eye(4)  # Rvv(p) covariance matrix
        v = np.random.multivariate_normal(np.zeros(4), Rvv)  # v(t) ~ N(0,Rvv(p))
        
        # y(t) = g(x(t),p) + v(t)
        y = h + v
        return y

    def output(self, x):
        """
        Output model: z(t) = h(x(t),p)
        
        Compute the deterministic outputs from the states (no measurement noise).
        
        Parameters
        ----------
        x : ndarray, shape (4,)
            Current states (mass in tanks)
            
        Returns
        -------
        z : ndarray, shape (2,)
            Deterministic outputs z(t) = h(x(t),p) (heights of first two tanks)
        """
        A = self.params[4:8]
        rho = self.params[11]
        
        # h(x(t),p): deterministic output function (mass to height, first two tanks only)
        heights = x / (rho * A)
        z = heights[:2]  # Only first two tanks as outputs
        return z