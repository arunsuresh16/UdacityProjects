
import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1
MAX_THROTTLE_SPEED = 0.2

class Controller(object):
    def __init__(self, vehicle_mass, fuel_cap, brake_deadband, decel_lmt,
    	accel_lmt, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        self.yaw_ctrl = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0  # Minimum throttle value
        mx = MAX_THROTTLE_SPEED  # Maximum throttle value

        self.throttle_ctrl = PID(kp, ki, kd, mn, mx)

        tau = 0.5  # 1/(2*pi*tau) = cutoff frequency
        ts = 0.02  # sample time

        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_cap = fuel_cap
        self.brake_deadband = brake_deadband
        self.decel_lmt = decel_lmt
        self.accel_lmt = accel_lmt
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.last_vel = 0

    def control(self, current_vel, angular_vel, linear_vel, dbw_enabled):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_ctrl.reset()
            return 0., 0., 0.

        # As the received velocity values can be noisy, we use low pass filter to average out
        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_ctrl.get_steering(linear_vel, angular_vel, current_vel)
        vel_err = linear_vel - current_vel
        self.last_vel = current_vel

        curr_time = rospy.get_time()
        sample_time = curr_time - self.last_time
        self.last_time = curr_time
        throttle = self.throttle_ctrl.step(vel_err, sample_time)
        brake = 0
        
        # Below check is required to completely stop the car
        if linear_vel == 0. and current_vel < 0.1:
            rospy.logdebug("Stopping the car")
            throttle = 0
            brake = 700  # 700Nm torque required to completely stop the car
        # In order to decelerate slowly without a jerk, below check is required
        elif throttle < .1 and vel_err < 0:
            rospy.logdebug("Decelerating")
            throttle = 0
            # As brake takes only positive values in Nm, below calculation is required
            decel = max(vel_err, self.decel_lmt)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius

        # rospy.logdebug("Velocity Error: {0}".format(vel_err))
        # rospy.logdebug("Throttle: {0}, Brake: {1}, Steering: {2}".format(throttle, brake, steering))
        return throttle, brake, steering
