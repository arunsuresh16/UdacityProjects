#ifndef PID_H
#define PID_H

#include <vector>
#include <math.h>

#define CALIBRATE       0

#define PID_KP_INIT     0.1004
#define PID_KI_INIT     0.0001
#define PID_KD_INIT     2.85
//#define PID_KP_INIT     0.1
//#define PID_KI_INIT     0.001
//#define PID_KD_INIT     2.8

#define TOLERANCE       0.005
#define NUM_OF_ITER     250

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Update the PID coeffecients by Twiddle method
   */
  void UpdateCoeffecientsByTwiddle(double total_error, double &parameterToTune);

  /**
   * Get the steering angle. Run this only after updating the error
   * @param cte The current cross track error
   * @output The total PID error
   */
  double GetNewSteeringAngle(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

 private:
  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;

 public:
  /**
   * PID Coefficients
   */ 
  double Kp;
  double Ki;
  double Kd;
};

#endif  // PID_H
