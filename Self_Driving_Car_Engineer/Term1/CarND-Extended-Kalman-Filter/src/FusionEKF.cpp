#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //Measurement matrix - LIDAR
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
			  
  //Measurement matrix - RADAR
  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1; 
			  
  // set the acceleration noise components
  noise_ax = 9;
  noise_ay = 9;

  // create a 4D state vector, we don't know yet the values of the x state
  VectorXd x_in = VectorXd(4);
  x_in = VectorXd::Zero(4);

  // state covariance matrix P
  MatrixXd P_in = MatrixXd(4, 4);
  P_in << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

  // the initial transition matrix F_
  MatrixXd F_in = MatrixXd(4, 4);
  F_in <<   1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

  // measurement matrix
  MatrixXd H_in = MatrixXd(2, 4);
  H_in = H_laser_;

  // measurement covariance
  MatrixXd R_in = MatrixXd(2, 2);
  R_in << 0.0225, 0,
            0, 0.0225;

  // the initial process noise covariance matrix
  MatrixXd Q_in = MatrixXd(4, 4);
  Q_in = MatrixXd::Zero(4,4);

  // Initialize all the variables in ekf with the above values
  ekf_.Init(x_in, P_in, F_in, H_in, R_in, Q_in);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * Initialize the state vector_ with the first measurement.
     * and then create the covariance matrix.
     */

    // first measurement
    cout << "ProcessMeasurement: First reading" << endl;

    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

      float rho = measurement_pack.raw_measurements_[0];
  	  float phi = measurement_pack.raw_measurements_[1];
  	  float rho_dot = measurement_pack.raw_measurements_[2];
  	  float px = rho * cos(phi);
      if ( px < 0.0001 ) {
        px = 0.0001;
      }
  	  float py = rho * sin(phi);
      if ( py < 0.0001 ) {
        py = 0.0001;
      }
  	  float vx = rho_dot * cos(phi);
  	  float vy = rho_dot * sin(phi);
      ekf_.x_ << px, py, vx , vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // set the state with the initial location and zero velocity
      ekf_.x_ << measurement_pack.raw_measurements_[0], //px
              measurement_pack.raw_measurements_[1],  //py
              0, //vx
              0; //vy
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  // delta - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  if(dt > 0.001)
  {
    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;

    // Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
           0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
           dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
           0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

    // predict
    ekf_.Predict();
  }

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = MatrixXd(3, 3);
    ekf_.R_ = R_radar_;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = MatrixXd(3, 4);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = MatrixXd(2, 2);
    ekf_.R_ = R_laser_;
    ekf_.H_ = MatrixXd(2, 4);
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
