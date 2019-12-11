/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

#define NUM_OF_PARTICLES	100
//Uncomment the below to enable logging
//#define DEBUG 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
   if(is_initialized)
   {	   
      cout<<"Already Initialized"<<endl;
	  return;
   }
  num_particles = NUM_OF_PARTICLES;
    
  // Resize weights vector based on number of particles
  weights.resize(num_particles);
  
  std::default_random_engine random_gen;

  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < num_particles; ++i) {
    Particle temp_part = {};
	temp_part.id = i;
    temp_part.x = dist_x(random_gen);
    temp_part.y = dist_y(random_gen);
    temp_part.theta = dist_theta(random_gen);
	temp_part.weight = 1.0;
	particles.push_back(temp_part);
  }
  cout<<"Initialized "<<num_particles<<" number of particles"<<endl;
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   
  std::default_random_engine random_gen;
  
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);	 
	  
   for (auto &particle:particles)
   {
	  if (fabs(yaw_rate) <= 0.0001) { 
		particle.x += velocity * delta_t * cos(particle.theta);
		particle.y += velocity * delta_t * sin(particle.theta);
	  }
	  else
	  {
	    particle.x += (velocity * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta)))/yaw_rate;
	    particle.y += (velocity * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t))))/yaw_rate;
	    particle.theta += yaw_rate * delta_t;	   
	  }
	  
      particle.x += dist_x(random_gen);
      particle.y += dist_y(random_gen);
      particle.theta += dist_theta(random_gen);	
   }
#ifdef DEBUG
   cout<<"Done Prediction"<<endl;
#endif  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   double smallest_dist;
   for(auto &observation:observations)
   {
	   // Initializing to something big
	  smallest_dist = std::numeric_limits<double>::max();
      int id;
      for(auto &predict:predicted)
	  { 
	    double temp = dist(predict.x, predict.y, observation.x, observation.y);
		if(temp < smallest_dist)
		{
		  smallest_dist = temp;
		  id = predict.id;
		}		
	  }	 
	  observation.id = id;      	  
   }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
#ifdef DEBUG
   for(int j = 0; j <weights.size(); j++)
   {
     cout<<"Weight start"<<j<<" "<<std::fixed<<std::setprecision(5)<<weights[j]<<endl;
   }
#endif
   
   for (auto &particle:particles)
   {	   
     vector<LandmarkObs> predictions;
	 for (auto landmarks:map_landmarks.landmark_list)
     {
	   double l_x = (double)landmarks.x_f;
	   double l_y = (double)landmarks.y_f;
	   if(dist(l_x, l_y, particle.x, particle.y) <= sensor_range)
	   {
	     predictions.push_back(LandmarkObs{landmarks.id_i, l_x, l_y});	   
       }
	 }
	 
     // Transform and rotate to map co-ordinate
     vector<LandmarkObs> transformed_observations;
	 for (auto &observation:observations)
	 {
		 LandmarkObs transformed_observation;
         transformed_observation.x = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
         transformed_observation.y = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);	
         transformed_observation.id = observation.id; 
		 transformed_observations.push_back(transformed_observation);
	 }
	 
     dataAssociation(predictions, transformed_observations);
    
	 // reinit weight
     particle.weight = 1.0;
	
	 // Get the corresponding associated prediction for each transformed observation
	 for(auto &transformed_observation:transformed_observations)
	 {		 
		for(auto &prediction:predictions)
		{
		  if(prediction.id == transformed_observation.id)
		  {
	        particle.weight *= multiv_prob(std_landmark[0], std_landmark[1], transformed_observation.x, \
		    transformed_observation.y, prediction.x, prediction.y);
			break;
		  }
		}      
	 }
   }	

  for(unsigned int i=0; i < particles.size(); i++){
	weights[i] = particles[i].weight;
  }  
  
#ifdef DEBUG
   for(int j = 0; j <weights.size(); j++)
   {
     cout<<"Weight start"<<j<<" "<<std::fixed<<std::setprecision(5)<<weights[j]<<endl;
   }
#endif
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    std::default_random_engine random_gen;
	std::discrete_distribution<> weighted_distribution(weights.begin(),weights.end());
	std::vector<Particle> resampled_particles;

	for (unsigned int i = 0; i < particles.size() ; ++i) {
		int random_index = weighted_distribution(random_gen);
		resampled_particles.push_back(particles[random_index]);
	}

	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y; 
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}