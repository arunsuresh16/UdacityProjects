#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define PI 3.14159265

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  UpdateNewEstimate(y);
}

float KalmanFilter::normalizeAngle(float rad)
{
  while (rad <= -PI) rad += 2*PI;
  while (rad > PI) rad -= 2*PI;
  return rad;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */
   
  //Checking the position vector is not zero
  if(x_(0) == 0. && x_(1) == 0.)
    return;
  float rho_pred = sqrt(((x_(0) * x_(0)) +  (x_(1) * x_(1))));
  float phi_pred = atan2(x_(1), x_(0));
  // Normalize the angle here
  phi_pred = normalizeAngle(phi_pred);

  //Checking the value is not zero
  if (rho_pred < 0.0001) {
    rho_pred = 0.0001;
  } 
  float rho_dot_pred = ((x_(0) * x_(2)) + (x_(1) * x_(3))) / rho_pred;

  
  //Finding h(x)
  VectorXd hx = VectorXd(3);
  hx << rho_pred, phi_pred, rho_dot_pred;

  VectorXd y = z - hx;
  // Normalize the angle here
  y[1] = normalizeAngle(y[1]);
  // Linear  approximation of h(x) is now required. Here H_ is Hj - Jacobian Matrix
  UpdateNewEstimate(y);
}

void KalmanFilter::UpdateNewEstimate(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}