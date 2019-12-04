#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main(int argc, char** argv) {
  uWS::Hub h;

  PID pid;
  unsigned int num_of_iter = NUM_OF_ITER;
  unsigned int enable_twiddle = 0;
  std::string tune_parameter = "Kp";
  if(argc == 7)
  {
    double Kp_ = atof(argv[1]);
    double Ki_ = atof(argv[2]);
    double Kd_ = atof(argv[3]);
    enable_twiddle = atoi(argv[4]);
    num_of_iter = atoi(argv[5]);
    tune_parameter = argv[6];
    pid.Init(Kp_, Ki_, Kd_);
  }
  else
  {
    pid.Init(PID_KP_INIT, PID_KI_INIT, PID_KD_INIT);
  }

  h.onMessage([&num_of_iter, &enable_twiddle, &tune_parameter, &pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));


      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
//           double speed = std::stod(j[1]["speed"].get<string>());
//           double angle = std::stod(j[1]["steering_angle"].get<string>());

          static bool init_flag = true;
          // This is required to restart the simulator
          if(init_flag)
          {
            init_flag = false;
            std::string msg = "42[\"reset\",{}]";
            std::cout << "RESET: "<<msg << std::endl;
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          }
          if(enable_twiddle != 0)
          {
            static unsigned int iteration = 0;
            static double total_error = 0.0;
  //          std::cout<<"Iteration - "<<iteration<<std::endl;
            if(iteration == num_of_iter)
            {
              if(tune_parameter == "Kp")
              {
                std::cout << "Tuning Kp\n";
                pid.UpdateCoeffecientsByTwiddle(total_error, pid.Kp);
              }
              else if(tune_parameter == "Ki")
              {
                std::cout << "Tuning Ki\n";
                pid.UpdateCoeffecientsByTwiddle(total_error, pid.Ki);
              }
              else if(tune_parameter == "Kd")
              {
                std::cout << "Tuning Kd\n";
                pid.UpdateCoeffecientsByTwiddle(total_error, pid.Kd);
              }
              else
              {
                std::cout << "Tuning Kp\n";
                pid.UpdateCoeffecientsByTwiddle(total_error, pid.Kp);
              }
              iteration = 0;
              total_error = 0.0;
              std::string msg = "42[\"reset\",{}]";
              std::cout << "RESET: "<<msg << std::endl;
              ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
            else
            {
              total_error += pow(cte, 2);
            }
            iteration++;
          }
          double steer_value = pid.GetNewSteeringAngle(cte);
          /**
           * TODO: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */
          
          // DEBUG
//          std::cout << "CTE: " << cte << " Steering Value: " << steer_value
//                    << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
     //     std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
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
