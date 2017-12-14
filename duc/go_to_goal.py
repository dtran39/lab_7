#Samuel Conrad and Josh Jibilian

#!/usr/bin/env python3

import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

from cozmo.util import degrees, distance_mm, speed_mmps
from math import atan2
from cozmo.anim import Triggers

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"


async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
    cv2.imshow("Markers", opencv_image)

    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = []
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        # print('x =', x, 'y =', y,'theta =', yaw)
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list


#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    if cvt_inch:
        last_x, last_y = last_x / 25.6, last_y / 25.6
        curr_x, curr_y = curr_x / 25.6, curr_y / 25.6

    return [[last_x, last_y, last_h],[curr_x, curr_y, curr_h]]

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)


async def run(robot: cozmo.robot.Robot):
    global last_pose
    global grid, gui

    # start streaming
    robot.camera.image_stream_enabled = True

    #start particle filter
    pf = ParticleFilter(grid)

    ############################################################################
    ######################### YOUR CODE HERE####################################

    isFinished = False
    moveVal = 1
    while True:
        pf = ParticleFilter(grid)

        await robot.set_lift_height(0).wait_for_completed()
        await robot.set_head_angle(degrees(5)).wait_for_completed()

        #Obtain Odom Info
        curr_pose = robot.pose
        odom = compute_odometry(curr_pose)

        #Obtain list of currently seen markers and their poses
        markers = await image_processing(robot)
        marker2d_list = cvt_2Dmarker_measurements(markers)

        #Update particle filter
        estimate = pf.update(odom, marker2d_list)

        #Update particle filter gui
        gui.show_particles(pf.particles)
        gui.show_mean(estimate[0], estimate[1], estimate[2], estimate[3])
        gui.updated.set()

        while not estimate[3]:
            await robot.set_lift_height(0).wait_for_completed()
            await robot.set_head_angle(degrees(5)).wait_for_completed()

            #Obtain Odom Info
            curr_pose = robot.pose
            odom = compute_odometry(curr_pose)

            #Obtain list of currently seen markers and their poses
            markers = await image_processing(robot)
            marker2d_list = cvt_2Dmarker_measurements(markers)

            #Update particle filter
            estimate = pf.update(odom, marker2d_list)

            #Update particle filter gui
            gui.show_particles(pf.particles)
            gui.show_mean(estimate[0], estimate[1], estimate[2], estimate[3])
            gui.updated.set()

            last_pose = curr_pose

            if estimate[3]:
                continue

            numMarkers = len(markers)
            if moveVal == -1 or (numMarkers > 0 and marker2d_list[0][0] > 2.0):
                await robot.drive_straight(distance_mm(moveVal * 50), speed_mmps(50)).wait_for_completed()
                if moveVal == -1:
                    await robot.turn_in_place(degrees(30)).wait_for_completed()
                moveVal *= -1
            else:
                await robot.turn_in_place(degrees(30)).wait_for_completed()

        #Rotate toward goal
        x, y, h, c = compute_mean_pose(pf.particles)

        if x > 13:
            isDockingRegion = False
            goal = 19,9,0
            print("warehouse region")
        else:
            isDockingRegion = True
            goal = 9,9,0
            print("docking region")

        y_diff = goal[1] * 0.95 - y
        x_diff = goal[0] * 0.95 - x
        tan = math.degrees(atan2(y_diff, x_diff))
        rot = diff_heading_deg(tan, h)
        await robot.turn_in_place(degrees(rot)).wait_for_completed()

        #Move toward goal
        dist_to_goal = math.sqrt(y_diff**2 + x_diff**2) * 25
        dist = 0.0
        while dist < dist_to_goal:
            min_dist = min(30, dist_to_goal - dist)
            dist_mm = distance_mm(min_dist)
            await robot.drive_straight(dist_mm, speed_mmps(40)).wait_for_completed()
            dist = dist + min_dist


        goal_rot = -1 * tan
        await robot.turn_in_place(degrees(goal_rot)).wait_for_completed()
        if isDockingRegion:
            await robot.turn_in_place(degrees(145)).wait_for_completed()
        else:
            await robot.turn_in_place(degrees(180)).wait_for_completed()

        cube = None
        try:
            cube = await robot.world.wait_for_observed_light_cube(timeout=1)
        except:
            pass
        timer = 1
        numBoxesPlaced = 0
        previousPose = robot.pose
        while True:
            if timer == 2:
                await robot.turn_in_place(degrees(60)).wait_for_completed()
                try:
                    cube = await robot.world.wait_for_observed_light_cube(timeout=1)
                except:
                    pass
                timer = 0
            else:
                await robot.turn_in_place(degrees(-30)).wait_for_completed()
                try:
                    cube = await robot.world.wait_for_observed_light_cube(timeout=1)
                except:
                    pass
                timer += 1

            if cube:
                await robot.pickup_object(cube, num_retries=3).wait_for_completed()
                await robot.go_to_pose(previousPose).wait_for_completed()
                if isDockingRegion:
                    await robot.turn_in_place(degrees(-145)).wait_for_completed()
                    await robot.drive_straight(distance_mm(80), speed_mmps(40)).wait_for_completed()
                else:
                    if numBoxesPlaced == 0:
                        await robot.turn_in_place(degrees(-125)).wait_for_completed()
                        await robot.drive_straight(distance_mm(80), speed_mmps(40)).wait_for_completed()
                    else:
                        await robot.turn_in_place(degrees(-165)).wait_for_completed()
                        await robot.drive_straight(distance_mm(40), speed_mmps(40)).wait_for_completed()

                await robot.set_lift_height(0).wait_for_completed()
                await robot.go_to_pose(previousPose).wait_for_completed()
                cube = None
                timer = 1
                numBoxesPlaced += 1


    ############################################################################


class CozmoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()
