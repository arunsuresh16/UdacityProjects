#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h" // Required to smooth out the points without violating jerk or acceleration

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

void print_vector(string string_to_print, vector<double> to_print)
{
  std::cout << string_to_print << std::endl;
  for(auto i:to_print)
  {
    std::cout<<i<<"\n";
  }
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
	std::istringstream iss(line);
	double x;
	double y;
	float s;
	float d_x;
	float d_y;
	iss >> x;
	iss >> y;
	iss >> s;
	iss >> d_x;
	iss >> d_y;
	map_waypoints_x.push_back(x);
	map_waypoints_y.push_back(y);
	map_waypoints_s.push_back(s);
	map_waypoints_dx.push_back(d_x);
	map_waypoints_dy.push_back(d_y);
  }

  // New variables
  int lane = 1; // Start with middle lane as per simulator
  int count = 0;
  double ref_velocity = STARTING_SPEED; // Starting speed in mph

  h.onMessage([&count,&ref_velocity,&lane,&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
			   &map_waypoints_dx,&map_waypoints_dy]
			  (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
			   uWS::OpCode opCode) {

	// "42" at the start of the message means there's a websocket message event.
	// The 4 signifies a websocket message
	// The 2 signifies a websocket event
	if (length && length > 2 && data[0] == '4' && data[1] == '2') {

	  auto s = hasData(data);

	  if (s != "") {
		auto j = json::parse(s);
		
		string event = j[0].get<string>();
		
		if (event == "telemetry") {
		  // j[1] is the data JSON object
		  
		  // Main car's localization Data
		  double car_x = j[1]["x"];
		  double car_y = j[1]["y"];
		  double car_s = j[1]["s"];
		  double car_d = j[1]["d"];
		  double car_yaw = j[1]["yaw"];
		  double car_speed = j[1]["speed"];

		  // Previous path data given to the Planner
		  auto previous_path_x = j[1]["previous_path_x"];
		  auto previous_path_y = j[1]["previous_path_y"];
		  // Previous path's end s and d values 
		  double end_path_s = j[1]["end_path_s"];
		  double end_path_d = j[1]["end_path_d"];

		  // Sensor Fusion Data, a list of all other cars on the same side 
		  //   of the road.
		  auto sensor_fusion = j[1]["sensor_fusion"];

		  json msgJson;

		  vector<double> next_x_vals;
		  vector<double> next_y_vals;

		  int previous_path_size = previous_path_x.size(); // Previous list of points which could help in transition

		  // To avoid collision
		  if(previous_path_size > 0)
		  {
			  // Changing the car_s to represent previous path last point s
			  car_s = end_path_s;
		  }

		  // To change lane
		  bool too_close = false;
		  bool left_lane_car_too_close = false;
		  bool middle_lane_car_too_close = false;
		  bool right_lane_car_too_close = false;

		  // Prediction
		  // Find if any car is too close to slow down or lane change
		  for(auto single_sensor_fusion:sensor_fusion)
		  {
			double check_car_s  = single_sensor_fusion[5];
			double check_car_d  = single_sensor_fusion[6];
			double check_speed = getMagnitudeOfVector(single_sensor_fusion[3], single_sensor_fusion[4]);

			// If using previous points can project s value out. To look at where the car is in the future
			check_car_s += (double) (previous_path_size * check_speed * CAR_VISITING_TIME);

			// OR the flag as we will be going through a multiple sensor_fusion values
			left_lane_car_too_close |= checkLaneSafety(LEFTMOST_LANE, car_s,
					check_car_s, check_car_d);

			middle_lane_car_too_close |= checkLaneSafety(MIDDLE_LANE, car_s,
					check_car_s, check_car_d);

			right_lane_car_too_close |= checkLaneSafety(RIGHTMOST_LANE, car_s,
					check_car_s, check_car_d);
		  }

		  // Find if the car in front is close
		  if(lane == LEFTMOST_LANE)
		  {
			too_close = left_lane_car_too_close;
		  }
		  else if(lane == MIDDLE_LANE)
		  {
			too_close = middle_lane_car_too_close;
		  }
		  else if(lane == RIGHTMOST_LANE)
		  {
			too_close = right_lane_car_too_close;
		  }
		  // End of Prediction

		  // Behavior or action which our car needs to do
		  if(too_close)
		  {
			// Reduce the speed as there is a car in front
			ref_velocity -= SPEED_TO_DECREASE;

			if(lane == LEFTMOST_LANE)
			{
			  // Try to move to middle lane
			  if(!middle_lane_car_too_close)
			  {
				lane = MIDDLE_LANE;
				ref_velocity += (LANE_CHANGE_SPEED + SPEED_TO_DECREASE);
			  }
			}

			else if (lane == MIDDLE_LANE)
			{
			  // Try to move to the left most lane first
			  if(!left_lane_car_too_close)
			  {
				  lane = LEFTMOST_LANE;
				  ref_velocity += (LANE_CHANGE_SPEED + SPEED_TO_DECREASE);
			  }

			  // Try to move to the right most lane
			  if(!right_lane_car_too_close)
			  {
				  lane = RIGHTMOST_LANE;
				  ref_velocity += (LANE_CHANGE_SPEED + SPEED_TO_DECREASE);
			  }
			}

			else if (lane == RIGHTMOST_LANE)
			{
			  // Try to move to the middle lane
			  if(!middle_lane_car_too_close)
			  {
				  lane = MIDDLE_LANE;
				  ref_velocity += (LANE_CHANGE_SPEED + SPEED_TO_DECREASE);
			  }
			}
		  }

		  else if (ref_velocity < SPEED_LIMIT)
		  {
			ref_velocity += SPEED_TO_INCREASE;
		  }

		  // If on right lane, try to move to left most plane after a certain count
		  if((lane > 0) && (ref_velocity > LEFT_LANE_CHANGE_SPEED))
		  {
			count += 1;
			if(count > CHANGE_LEFT_COUNTER)
			{
			  std::cout << "Trying to change to left" << std::endl;
			  count = 0;

			  if(lane == 1)
			  {
				if(!left_lane_car_too_close)
				{
				  lane = LEFTMOST_LANE;
				  ref_velocity += LANE_CHANGE_SPEED;
				}
			  }
			  else if (lane == 2)
			  {
				if(!middle_lane_car_too_close)
				{
				  lane = MIDDLE_LANE;
				  ref_velocity += LANE_CHANGE_SPEED;
				}
			  }
			}
		  }
		  // End of Behavior

		  // Limit the speed to max speed limit
		  limitSpeed(ref_velocity);

		  // Generate Trajectory and Smoothen the path
		  // Below are all points required to interpolate with a spline
		  vector<double> points_x; // Anchor points
		  vector<double> points_y;

		  // Reference points and yaw
		  double reference_x = car_x;
		  double reference_y = car_y;
		  double reference_yaw = deg2rad(car_yaw);

		  // Getting 2 points atleast from the previous path
		  // When there are no previous path or the car just started moving, we need the cars position as the reference
		  if(previous_path_size < 2)
		  {
			// Going back in time to find the previous x and y
			double previous_x = car_x - cos(car_yaw);
			double previous_y = car_y - sin(car_yaw);

			// use the points that make the path tangent to the car
			points_x.push_back(previous_x);
			points_y.push_back(previous_y);

			points_x.push_back(car_x);
			points_y.push_back(car_y);
		  }
		  else
		  {
			// New reference x and y will be the last point of previous path
			reference_x = previous_path_x[previous_path_size - 1];
			reference_y = previous_path_y[previous_path_size - 1];

			double reference_previous_x = previous_path_x[previous_path_size - 2];
			double reference_previous_y = previous_path_y[previous_path_size - 2];
			points_x.push_back(reference_previous_x);
			points_y.push_back(reference_previous_y);

			points_x.push_back(reference_x);
			points_y.push_back(reference_y);

			reference_yaw  = atan2(reference_y - reference_previous_y, reference_x - reference_previous_x);
		  }

		  vector<double> next_waypointXY0 = getXY(car_s + (SPACING_POINTS * 1), (2 + (4 * lane)), map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  points_x.push_back(next_waypointXY0[0]);
		  points_y.push_back(next_waypointXY0[1]);

		  vector<double> next_waypointXY1 = getXY(car_s + (SPACING_POINTS * 2), (2 + (4 * lane)), map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  points_x.push_back(next_waypointXY1[0]);
		  points_y.push_back(next_waypointXY1[1]);

		  vector<double> next_waypointXY2 = getXY(car_s + (SPACING_POINTS * 3), (2 + (4 * lane)), map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  points_x.push_back(next_waypointXY2[0]);
		  points_y.push_back(next_waypointXY2[1]);

		  for(int i = 0; i < points_x.size(); i++)
		  {
			// Shift car reference angle to 0 degrees
			double shift_x = points_x[i] - reference_x;
			double shift_y = points_y[i] - reference_y;

			// Transform and rotate to local coordinates
			points_x[i] = (shift_x * cos(0 - reference_yaw)) - (shift_y * sin(0 - reference_yaw));
			points_y[i] = (shift_x * sin(0 - reference_yaw)) + (shift_y * cos(0 - reference_yaw));
		  }

		  // Create a spline
		  tk::spline s;

		  // Set all the points onto this spline
		  s.set_points(points_x, points_y);

		  // Add the previous path as this will help in transition
		  for(int i = 0; i < previous_path_size; i++)
		  {
			next_x_vals.push_back(previous_path_x[i]);
			next_y_vals.push_back(previous_path_y[i]);
		  }

		  // Calculate how to space the spline points so that the car travels at desired velocity
		  double target_x = SPACING_POINTS;
		  double target_y = s(target_x);
		  double target_distance = getMagnitudeOfVector(target_x, target_y); // As this is the hypotenuse of a right angle triangle
		  double x_add_on = 0;
		  // vel is in mph, converting it to mtrs per sec with 2.24
		  double N = target_distance / (CAR_VISITING_TIME * ref_velocity / 2.24);

		  for(int i = 0; i <= NUM_OF_POINTS - previous_path_size; i++)
		  {
			double x = x_add_on + (target_x / N);
			double y = s(x);

			x_add_on = x;

			double temp_x = x;
			double temp_y = y;
			// Transform and rotate back to global coordinates
			x = (temp_x * cos(reference_yaw)) - (temp_y * sin(reference_yaw));
			y = (temp_x * sin(reference_yaw)) + (temp_y * cos(reference_yaw));
			x += reference_x;
			y += reference_y;

			next_x_vals.push_back(x);
			next_y_vals.push_back(y);
		  }

#if ENABLE_DEBUG
		  print_vector("next_x_vals", next_x_vals);
		  print_vector("next_y_vals", next_y_vals);
#endif

		  msgJson["next_x"] = next_x_vals;
		  msgJson["next_y"] = next_y_vals;

		  auto msg = "42[\"control\","+ msgJson.dump()+"]";

		  ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
		}  // end "telemetry" if
	  } else {
		// Manual driving
		std::string msg = "42[\"manual\",{}]";
		ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
	  }
	}  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
	std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
						 char *message, size_t length) {
	ws.close();
	std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
	std::cout << "Listening to port " << port << std::endl;
  } else {
	std::cerr << "Failed to listen to port" << std::endl;
	return -1;
  }
  
  h.run();
}
