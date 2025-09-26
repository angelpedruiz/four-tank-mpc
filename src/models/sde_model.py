import numpy as np

class FourTankSDE:
    """
    Stochastic model of the Modified Four Tank System. 
    The disturbances are modeled using a stochastic disturbance model.
    There is also measurement noise.

    States: mass of liquid in tanks [m1, m2, m3, m4] [g]
    Manipulated variables: pump flows [F1, F2] [cm^3/s]
    Disturbance variables: unmeasured flow rates [F3, F4] [cm^3/s]
    Parameters: pipe areas, tank areas, flow splits, gravity, density
    """

    def __init__(self, params: np.ndarray, measurement_noise_std: float, disturbance_noise_std: float, x0: np.ndarray = None):
        """
        Initialize the four-tank SDE system.

        Parameters
        ----------
        params : ndarray, shape (12,)
            System parameters:
                params[0:4]   -> pipe cross-sectional areas a [cm^2]
                params[4:8]   -> tank cross-sectional areas A [cm^2]
                params[8:10]  -> flow distribution ratios gamma [-]
                params[10]    -> gravity g [cm/s^2]
                params[11]    -> density rho [g/cm^3]
        measurement_noise_std : float
            Standard deviation for measurement noise v(t) [cm]
        disturbance_noise_std : float
            Standard deviation (sigma) for Brownian motion disturbances [cm^3/s]
        x0 : ndarray, shape (4,), optional
            Initial states (mass in each tank). Defaults to zeros.
        """
        self.params = params
        self.x0 = x0 if x0 is not None else np.zeros(4)
        self.measurement_noise_std = measurement_noise_std
        self.disturbance_noise_std = disturbance_noise_std

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        SDE dynamics: dx(t) = f(x(t),u(t),d(t),p)dt + σ(x(t),u(t),d(t),p)dω(t)
        
        Parameters
        ----------
        t : float
            Current time
        x : ndarray, shape (4,)
            Current states (mass in tanks)
        u : ndarray, shape (2,)
            Manipulated variables (pump flows)
        dt : float
            Time step for Brownian motion generation

        Returns
        -------
        dx : ndarray, shape (4,)
            State increment dx = f*dt + σ*dω
        """
        # Deterministic drift term: f(x(t),u(t),d(t),p)
        drift = self._drift_term(x, u)
        
        # Generate Brownian motion increments: dω ~ N(0, dt*I)
        dw = np.random.normal(0, np.sqrt(dt), size=2)
        
        # Stochastic diffusion term: σ(x(t),u(t),d(t),p)dω(t)
        diffusion = self._diffusion_term(x, u, dw)
        
        # SDE: dx(t) = f*dt + σ*dω
        dx = drift * dt + diffusion
        
        return dx, diffusion
    
    def _drift_term(self, x, u):
        """
        Deterministic drift term f(x(t),u(t),d(t),p)
        """
        a = self.params[:4]
        A = self.params[4:8]
        gamma = self.params[8:10]
        g = self.params[10]
        rho = self.params[11]
        
        # Inflows (without stochastic disturbances)
        qin = np.zeros(4)
        qin[0] = gamma[0] * u[0]
        qin[1] = gamma[1] * u[1]
        qin[2] = (1 - gamma[1]) * u[1]
        qin[3] = (1 - gamma[0]) * u[0]

        # Heights
        h = x / (rho * A)

        # Outflows
        qout = a * np.sqrt(2 * g * h)

        # Mass balances (deterministic part)
        f = np.zeros(4)
        f[0] = rho * (qin[0] + qout[2] - qout[0])
        f[1] = rho * (qin[1] + qout[3] - qout[1])
        f[2] = rho * (qin[2] - qout[2])
        f[3] = rho * (qin[3] - qout[3])

        return f
    
    def _diffusion_term(self, x, u, dw):
        """
        Stochastic diffusion term σ(x(t),u(t),d(t),p)dω(t)
        
        Models F3 and F4 disturbances as Brownian motion.
        """
        rho = self.params[11]
        
        # Diffusion matrix σ(x,u,d,p)
        # F3 affects tank 3, F4 affects tank 4
        sigma = np.zeros((4, 2))
        sigma[2, 0] = rho * self.disturbance_noise_std  # F3 -> tank 3
        sigma[3, 1] = rho * self.disturbance_noise_std  # F4 -> tank 4
        
        # σ(x,u,d,p) * dω(t)
        diffusion = sigma @ dw
        
        return diffusion
    
    def measurement(self, x):
        """
        Compute the measurements from the states. Includes measurement noise.

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
        
        # v(t) ~ N(0, Rvv(p)): measurement noise
        Rvv = (self.measurement_noise_std ** 2) * np.eye(4)
        v = np.random.multivariate_normal(np.zeros(4), Rvv)
        y = h + v
        return y

    def output(self, x):
        """
        Output model: z(t) = h(x(t),p)
        
        Deterministic outputs (no measurement noise).
        """
        A = self.params[4:8]
        rho = self.params[11]
        
        # h(x(t),p): deterministic output function
        heights = x / (rho * A)
        z = heights[:2]  # Only first two tanks as outputs
        return z

    def get_initial_state(self):
        """Return the initial state of the system."""
        return self.x0
