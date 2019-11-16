#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
STOPLINE_BACK_WPS = 5  # Number of waypoints back from the stop line so that the front of the car is also considered
USE_TRAFFIC_LIGHT_DETECTION_INFO = 1
MAX_DECEL = 1.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_wp_idx = -1
        self.base_lane = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x  # Getting the current position of the car
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x,y], 1)[1]  # KDTree way to get the index of first closest point to x, y
        # check if closest is ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for Hyperplane
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if USE_TRAFFIC_LIGHT_DETECTION_INFO:
            # Return the base waypoints in case there is no light detected or
            # the light detected is farther than the LOOKAHEAD_WPS
            if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
                lane.waypoints = base_waypoints
            else:
                # rospy.logdebug("Red light detected, decelerating..")
                lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        else:
            lane.waypoints = base_waypoints

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        # To not overwrite the only base waypoint available
        temp = []
        for i,wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - STOPLINE_BACK_WPS, 0)
            # Calculate the distance as to how far away from the light we need to stop
            dist = self.distance(waypoints, i, stop_idx)
            # The below needs to change to avoids steep decelretate
            vel = math.sqrt(2*MAX_DECEL*dist) #math.sqrt(2*MAX_DECEL*dist)
            # If velocity is small enough, then velocity can be made as 0
            if vel < 1.:
                vel = 0
            # Consider either the speed limit or the decelerated velocity whichever is small
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x )
            temp.append(p)

        return temp
    
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        rospy.logwarn('Received Base Waypoints')
        self.base_waypoints = waypoints  # Make a copy of the waypoints as its received only once

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            # Use KDTree to efficiently traverse through the waypoints to find the closest
            # LOOKAHEAD_WPS number of waypoints in  logn
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
