import numpy as np
from scipy.signal import lti

class LinearStateSystem():
    def __init__(self, A, B, C, D, x0):
        self.state = x0
        self.system = lti(A, B, C, D)
    
    @property
    def state_dim(self):
        return self.system.A.shape[1]
    
    @property
    def input_dim(self):
        return self.system.B.shape[1]

    @property
    def output_dim(self):
        return self.system.C.shape[0]
    
    @property
    def output(self):
        return self.system.C @ self.state
    
    def step(self, control, dt):
        self.state = self.system.to_discrete(dt).A@self.state + self.system.to_discrete(dt).B@control


class TimeVaryingLinearStateSystem(LinearStateSystem):
    def __init__(self, lambdaA, lambdaB, lambdaC, lambdaD, theta0, x0):
        self.theta = theta0
        self.state = x0
        self._A, self._B, self._C, self._D = lambdaA, lambdaB, lambdaC, lambdaD
        
    @property
    def system(self):
        return lti(self._A(self), self._B(self), self._C(self), self._D(self))   

    def change_theta(self, new_theta):
        self.theta = new_theta


class NoisySystem(TimeVaryingLinearStateSystem):
    def __init__(self, *args, zetax, zetay, zetatheta):
        print(args)
        super().__init__(*args)
        self.zetax, self.zetay, self.zetatheta = zetax, zetay, zetatheta
    
    @property
    def output(self):
        return super().output + self.zetay(self.state)
    
    def step(self, control, dt):
        super().step(control, dt)
        self.state = self.state + self.zetax(self.state, dt)
        self.theta = self.theta + self.zetatheta(self.theta, dt)


class CarSystem(TimeVaryingLinearStateSystem):
    def __init__(self, start_position=0, start_velocity=0, mass_kg=1000, friction_coeff=10, slope_degr=5, motor_gain=1):
        slope_rad = slope_degr*np.pi/180
        disturbance = 9.81*np.sin(slope_rad)*mass_kg

        state = np.vstack([start_velocity,start_position,1])
        theta = np.vstack([mass_kg, friction_coeff, disturbance, motor_gain])

        A = lambda system: np.array([
                [-system.friction_coeff/system.mass_kg, 0., -system.disturbance/system.mass_kg],
                [1., 0., 0.], 
                [0., 0., 0.]
        ])
        B = lambda system: np.array([
                [system.motor_gain/system.mass_kg,],
                [0.],
                [0.]
            ])
        C = lambda system: np.array([[0., 1., 0.]])
        D = lambda system: np.array([[0.]])

        super().__init__(A, B, C, D, theta, state)

    @property
    def mass_kg(self):
        return self.theta[0].item()
    
    @property
    def friction_coeff(self):
        return self.theta[1].item()
    
    @property
    def disturbance(self):
        return self.theta[2].item()
    
    @property
    def motor_gain(self):
        return self.theta[3].item()
    
    @property
    def position(self):
        return self.state[1].item()
    
    @property
    def velocity(self):
        return self.state[0].item()
    

class SystemLogger():
    def __init__(self):
        self.statelog = []
        self.paramlog = []

    def log(self, system):
        self.statelog.append(system.state)
        self.paramlog.append(system.theta)

    @property
    def state_trajectories(self):
        return np.hstack(self.statelog)
    
    @property
    def parameters_trajectories(self):
        return np.hstack(self.paramlog)