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

    def __init__(self, params: np.ndarray, measurement_noise_std: float, disturbance_noise_std: float, correlation_time: float = 1.0, x0: np.ndarray = None):
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
            Intensity (epsilon) of stochastic flow disturbances [cm^3/s]
        correlation_time : float, optional
            Correlation time (tau) for the stochastic disturbances [s]. Default: 1.0
        x0 : ndarray, shape (4,), optional
            Initial states (mass in each tank). Defaults to zeros.
        """
        self.params = params
        self.x0 = x0 if x0 is not None else np.zeros(4)
        self.measurement_noise_std = measurement_noise_std
        self.disturbance_noise_std = disturbance_noise_std
        self.correlation_time = correlation_time

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, d: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        SDE dynamics: dx(t) = f(x(t),u(t),d(t),p)dt + ρ*ε*√(τ*dt)*ξ

        Parameters
        ----------
        t : float
            Current time
        x : ndarray, shape (4,)
            Current states (mass in tanks) [g]
        u : ndarray, shape (2,)
            Manipulated variables (pump flows) [cm³/s]
        d : ndarray, shape (2,)
            Deterministic disturbance variables [cm³/s]
        dt : float
            Time step [s]

        Returns
        -------
        dx : ndarray, shape (4,)
            State increment [g]: dx = f*dt + ρ*ε*√(τ*dt)*ξ
        diffusion : ndarray, shape (4,)
            Stochastic diffusion term [g]: ρ*ε*√(τ*dt)*ξ
        """
        # Deterministic drift term: f(x(t),u(t),d(t),p) [g/s]
        drift = self._drift_term(x, u, d)

        # Generate standard normal random variables: ξ ~ N(0, 1)
        xi = np.random.randn(2)

        # Stochastic diffusion term: ρ*ε*√(τ*dt)*ξ [g]
        diffusion = self._diffusion_term(x, u, dt, xi)

        # SDE: dx(t) = f*dt + ρ*ε*√(τ*dt)*ξ
        dx = drift * dt + diffusion

        return dx, diffusion
    
    def _drift_term(self, x, u, d):
        """
        Deterministic drift term f(x(t),u(t),d(t),p)

        Returns: f with units [g/s] (mass rate)
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
        # All terms converted to [g/s] by multiplying flow rates [cm³/s] by density rho [g/cm³]
        f = np.zeros(4)
        f[0] = rho * (qin[0] + qout[2] - qout[0])
        f[1] = rho * (qin[1] + qout[3] - qout[1])
        f[2] = rho * (qin[2] - qout[2] + d[0])  # d[0] is deterministic disturbance [cm³/s]
        f[3] = rho * (qin[3] - qout[3] + d[1])  # d[1] is deterministic disturbance [cm³/s]

        return f
    
    def _diffusion_term(self, x, u, dt, xi):
        """
        Stochastic diffusion term: ρ*ε*√(τ*dt)*ξ

        Models F3 and F4 disturbances as Brownian motion with correlation time.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Current states [g]
        u : ndarray, shape (2,)
            Control inputs [cm³/s]
        dt : float
            Time step [s]
        xi : ndarray, shape (2,)
            Standard normal random variables ~ N(0, 1) (dimensionless)

        Returns
        -------
        diffusion : ndarray, shape (4,)
            Diffusion term with units [g]

        Notes
        -----
        Units: ρ [g/cm³] * ε [cm³/s] * √(τ [s] * dt [s]) * ξ [-]
             = [g/cm³] * [cm³/s] * [s] * [-]
             = [g]
        """
        rho = self.params[11]

        # Diffusion coefficient: σ = ρ * ε * √(τ * dt)
        # Units: [g/cm³] * [cm³/s] * √([s]*[s]) = [g]
        sigma = rho * self.disturbance_noise_std * np.sqrt(self.correlation_time * dt)

        # Construct diffusion matrix (only tanks 3 and 4 have stochastic disturbances)
        diffusion = np.zeros(4)
        diffusion[2] = sigma * xi[0]  # F3 -> tank 3
        diffusion[3] = sigma * xi[1]  # F4 -> tank 4

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
