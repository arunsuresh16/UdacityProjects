#include "PID.h"
#include <iostream>

#include <math.h>
/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
	// Initializing all errors
  p_error = i_error = d_error = 0.0;

  // Initializing all errors
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

  std:: cout << "Initialized Kp with " << Kp << ", Ki with "
      << Ki << ", Kd with " << Kd <<std::endl;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  double prev_cte = p_error;
  p_error = cte;
  i_error += cte;
  d_error = cte - prev_cte;
}

void PID::UpdateCoeffecientsByTwiddle(double total_error, double &parameterToTune) {
  static bool init_flag = true;
  static bool first_reset_flag = true;
  static bool second_reset_flag = true;
  static double diffParams = 0.5;
  static double best_error = 100.0;

  if(init_flag)
  {
    std::cout << "Twiddle Init" <<std::endl;
    best_error = total_error;
    std::cout<< "After Kp = " << Kp << ", Ki = "<<Ki << ", Kd = "<<Kd << "total_error = " << total_error << "best_error " << best_error << std::endl;
    init_flag = false;
    return;
  }
  if(fabs(diffParams) > TOLERANCE)
  {
    if(first_reset_flag)
    {
      parameterToTune += diffParams;
      first_reset_flag = false;
      std::cout<< "After Kp = " << Kp << ", Ki = "<<Ki << ", Kd = "<<Kd << "total_error = " << total_error << "best_error " << best_error << std::endl;
      std::cout << "First reset" << std::endl;
      return;
    }

    std::cout<<"Err = "<<total_error<<", BestErr = "<<best_error<<std::endl;
    if (total_error < best_error)
    {
      best_error = total_error;
      diffParams *= 1.1;
      std::cout<<"Error got decreased, so increase "
          "diffParams to "<<diffParams << std::endl;
    }
    else
    {
      if(second_reset_flag)
      {
        parameterToTune -= 2 * diffParams;
        std::cout << "Second reset" << std::endl;
        std::cout<< "After Kp = " << Kp  << ", Ki = "<<Ki << ", Kd = "<<Kd << "total_error = " << total_error << "best_error " << best_error << std::endl;
        second_reset_flag = false;
        return;
      }
      parameterToTune += diffParams;
      diffParams *= 0.9;
      std::cout<<"Error got increased, so decrease "
          "diffParams to "<<diffParams << std::endl;
    }
    first_reset_flag = second_reset_flag = true;
  }
  else
  {
    std::cout << "**************************Done Tuning the parameter with final value - "
        << parameterToTune << std::endl;
  }

  std::cout<< "After Kp = " << Kp << ", Ki = "<<Ki << ", Kd = "<<Kd
      <<", diffParams = "<<diffParams<<std::endl;
}

double PID::GetNewSteeringAngle(double cte) {
  // First update the PID errors with the new CTE
  UpdateError(cte);
  double steer_value = TotalError();

  return steer_value;
}

double PID::TotalError(void) {
  /**
   * TODO: Calculate and return the total error
   */
  return ((-Kp * p_error) - (Ki * i_error) - (Kd * d_error));
}
