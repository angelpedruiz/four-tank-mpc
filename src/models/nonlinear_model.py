import numpy as np

class FourTankNonlinear:
    """
    Deterministic nonlinear model of the Modified Four Tank System.

    States: mass of liquid in tanks [m1, m2, m3, m4] [g]
    Manipulated variables: pump flows [F1, F2] [cm^3/s]
    Disturbance variables: unmeasured flow rates [F3, F4] [cm^3/s]
    Parameters: pipe areas, tank areas, flow splits, gravity, density
    """

    def __init__(self, params: dict, x0=None, dt: float=0.01):
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
        """
        self.params = params
        self.x0 = x0 if x0 is not None else np.zeros(4)
        self.dt = dt

    def dynamics(self, x: np.ndarray, u: np.ndarray, d: np.ndarray=None) -> np.ndarray:
        """
        Compute the derivatives of the system states. dx = f(x,u,d,p)dt

        Parameters
        ----------
        t : float
            Current time (required by ODE solvers)
        x : ndarray, shape (4,)
            Current states (mass in tanks)
        u : ndarray, shape (2,)
            Manipulated variables (pump flows)
        d: ndarray, shape (2,), optional
            Disturbance variables (unmeasured flows). Defaults to zeros.

        Returns
        -------
        xdot : ndarray, shape (4,)
            Time derivatives of the states
        """
        a = self.params[:4]
        A = self.params[4:8]
        gamma = self.params[8:10]
        g = self.params[10]
        rho = self.params[11]

        # Inflows
        qin = np.zeros(4)
        qin[0] = gamma[0] * u[0]
        qin[1] = gamma[1] * u[1]
        qin[2] = (1 - gamma[1]) * u[1]
        qin[3] = (1 - gamma[0]) * u[0]
        
        # Add disturbances if provided
        if d is not None:
            qin[2] += d[0] # F3
            qin[3] += d[1] # F4

        # Heights
        h = x / (rho * A)

        # Outflows
        qout = a * np.sqrt(2 * g * h)

        # Mass balances
        xdot = np.zeros(4)
        xdot[0] = rho * (qin[0] + qout[2] - qout[0])
        xdot[1] = rho * (qin[1] + qout[3] - qout[1])
        xdot[2] = rho * (qin[2] - qout[2])
        xdot[3] = rho * (qin[3] - qout[3])

        return xdot
    
    def measurement(self, x):
        """
        Compute the measurements from the states. y = g(x,p)

        Parameters
        ----------
        x : ndarray, shape (4,)
            Current states (mass in tanks)

        Returns
        -------
        y : ndarray, shape (4,)
            Measurements (heights in tanks)
        """
        A = self.params[4:8]
        rho = self.params[11]
        h = x / (rho * A)
        return h
    
    def output(self, x):
        """
        Compute the outputs from the states. z = h(x,p)
        """
        outputs = self.measurement(x)[:2]  # Heights of the first two tanks
        return outputs

