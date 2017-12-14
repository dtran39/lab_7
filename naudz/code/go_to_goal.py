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

from math import *

try:
    from PIL import ImageDraw, ImageFont, Image
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')


# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 4.0

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
    global state


    await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()

    # start streaming
    robot.camera.image_stream_enabled = True

    #start particle filter
    pf = ParticleFilter(grid)
    pf2 = ParticleFilter(grid)

    ############################################################################
    ######################### YOUR CODE HERE ###################################
    isStorage = True

    first_turn = True
    state = "GLP"
    m_confident = False

    pickup_wait = (4, 6.5, 90)
    storage_wait = (21, 12, 160)
    deliver_wait = None
    pickup_deliver = (8, 9, 0)
    storage_deliver = [(24, 6, 0), (24, 9, 0), (24, 12, 0)]
    gone_to_pickup = False
    cube_seen = False
    turn_left= False
    stored_package_count = 0
    first_stored_cube = None


    while(True):
        
        while (state == "GLP"):
            await robot.wait_for_all_actions_completed()
            await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
            await robot.set_lift_height(height = 0.0).wait_for_completed()

            GoneToPose = False
            m_confident = False

            odometry = compute_odometry(robot.pose)
            markers = await image_processing(robot)
            markers_poses = cvt_2Dmarker_measurements(markers)

            print(len(markers_poses))

            posX, posY, angle, m_confident = pf.update(odometry, markers_poses)

            print(posX, posY, angle)
            last_pose = robot.pose

            if not m_confident:
                await robot.drive_wheels(-10, 10)
            gui.show_particles(pf.particles)
            gui.show_mean(posX,posY,angle,m_confident)
            gui.updated.set()

            if m_confident:
                if posX <= 13.0:
                    state = "Pickup"
                else:
                    state = "Storage"
                    isStorage = True
                first_turn = True
                print("Changing State from GLP to {}".format(state))

        while (state == "Pickup"):
            await robot.wait_for_all_actions_completed()

            if not gone_to_pickup:
                await robot.wait_for_all_actions_completed()
                robot.stop_all_motors()

                newX = pickup_wait[0] - posX
                newY = pickup_wait[1] - posY
                dist = grid_distance(posX, posY, pickup_wait[0], pickup_wait[1]) * 25.6
                angle_rotate = atan2(newY, newX)
                angle_rotate_degrees = degrees(angle_rotate)
                diffheading = diff_heading_deg(angle_rotate_degrees, angle)
                await robot.turn_in_place(cozmo.util.degrees(diffheading)).wait_for_completed()

                drive_time = 3.0
                speed = dist / drive_time
                await robot.drive_wheels(speed, speed, duration=drive_time)

                gone_to_pickup = True
                await robot.turn_in_place(
                    cozmo.util.degrees(-angle_rotate_degrees + pickup_wait[2])).wait_for_completed()
                deliver_wait = pickup_deliver
                state = "Pickup Cube"

        while (state == "Pickup Cube"):
            
            print("looking for cube")

            await robot.wait_for_all_actions_completed()
            await robot.set_head_angle(cozmo.util.degrees(0.0)).wait_for_completed()
            await robot.set_lift_height(height = 0.0).wait_for_completed()

            try:
                ar_cube = await robot.world.wait_for_observed_light_cube(timeout=1)
                dst = (robot.pose.position.x - ar_cube.pose.position.x) ** 2 + (robot.pose.position.y - ar_cube.pose.position.y) ** 2
                if dst >= 70000:
                    print("TOO FAR")
                    ar_cube = None

            except:
                ar_cube = None


            if not ar_cube:
                if first_turn:
                    await robot.turn_in_place(cozmo.util.degrees(-15)).wait_for_completed()
                    first_turn = False
                direction = -1 if turn_left else 1
                await robot.turn_in_place(cozmo.util.degrees(direction * 40)).wait_for_completed()
                turn_left = not turn_left
            else:
                print ("I can see cube!")
                gone_to_pickup = False
                if not isStorage:
                    # robot.dock_with_cube(ar_cube, approach_angle=cozmo.util.degrees(0),
                    #                             alignment_type=cozmo.robot_alignment.RobotAlignmentTypes.Body)
                    await robot.pickup_object(ar_cube, num_retries=10).wait_for_completed()
                    await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
                    await robot.wait_for_all_actions_completed()
                    await robot.drive_wheels(40, 40, duration=6.2)
                    await robot.wait_for_all_actions_completed()
                    await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
                    await robot.wait_for_all_actions_completed()
                    await robot.drive_wheels(40, 40, duration=1)
                    state = "Pickup Home"
                else:
                    time.sleep(10)
                    await robot.pickup_object(ar_cube, num_retries=10).wait_for_completed()
                    if first_stored_cube:
                        await robot.set_head_angle(cozmo.util.degrees(0.0)).wait_for_completed()
                        cube_seen = None
                        while not cube_seen:
                            await robot.wait_for_all_actions_completed()
                            try:
                                cube_seen = await robot.world.wait_for_observed_light_cube(timeout=1)
                                if cube_seen != first_stored_cube:
                                    cube_seen = None
                                    await robot.drive_wheels(10, -10)
                                else:
                                    robot.stop_all_motors()
                            except:
                                await robot.drive_wheels(10, -10)

                        await robot.wait_for_all_actions_completed()
                        await robot.drive_wheels(30, 30, duration=3.0)
                        await robot.wait_for_all_actions_completed()
                        robot.place_on_object(first_stored_cube, num_retries=3).wait_for_completed()
                        await robot.wait_for_all_actions_completed()
                    else:
                        first_stored_cube = ar_cube
                        await robot.turn_in_place(cozmo.util.degrees(180)).wait_for_completed()
                        await robot.wait_for_all_actions_completed()
                        await robot.drive_wheels(40, 40, duration=9)
                    state = "Storage Home"

        while (state == "Pickup Home"):
            await robot.wait_for_all_actions_completed()
            await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
            await robot.set_lift_height(height = 0.0).wait_for_completed()

            await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
            await robot.wait_for_all_actions_completed()
            await robot.drive_wheels(60, 60, duration=3.0)
            await robot.wait_for_all_actions_completed()
            await robot.turn_in_place(cozmo.util.degrees(90)).wait_for_completed()
            await robot.wait_for_all_actions_completed()
            await robot.drive_wheels(40, 40, duration=4.0)
            await robot.wait_for_all_actions_completed()
            await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
            await robot.wait_for_all_actions_completed()
            await robot.drive_wheels(40, 40, duration=4.0)
            await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
            await robot.wait_for_all_actions_completed()

            state = "Pickup Cube"

        while (state == "Storage Home"):
            await robot.wait_for_all_actions_completed()
            await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
            await robot.set_lift_height(height = 0.0).wait_for_completed()

            await robot.turn_in_place(cozmo.util.degrees(180)).wait_for_completed()
            await robot.wait_for_all_actions_completed()
            await robot.drive_wheels(60, 60, duration=2.0)
            await robot.wait_for_all_actions_completed()
            state = "Pickup Cube"


        while (state == "Deliver Cube"):
            #RELOCALIZE

            await robot.wait_for_all_actions_completed()
            await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
            m_confident = False

            odometry = compute_odometry(robot.pose)
            markers = await image_processing(robot)
            markers_poses = cvt_2Dmarker_measurements(markers)

            print(len(markers_poses))

            posX, posY, angle, m_confident = pf.update(odometry, markers_poses)

            print(posX, posY, angle)
            last_pose = robot.pose

            if not m_confident:
                await robot.drive_wheels(-10, 10)
            gui.show_particles(pf.particles)
            gui.show_mean(posX,posY,angle,m_confident)
            gui.updated.set()

            #DRIVE TO RELAY

            if m_confident:
                await robot.wait_for_all_actions_completed()
                robot.stop_all_motors()

                newX = deliver_wait[0] - posX
                newY = deliver_wait[1] - posY
                dist = grid_distance(posX, posY, deliver_wait[0], deliver_wait[1]) * 25.6
                angle_rotate = atan2(newY, newX)
                angle_rotate_degrees = degrees(angle_rotate)
                diffheading = diff_heading_deg(angle_rotate_degrees, angle)
                #print(angle, otherAngleDegrees, diffheading)
                await robot.turn_in_place(cozmo.util.degrees(diffheading)).wait_for_completed()

                count = 0
                while (count < dist):
                    print("Dist Left : ", dist - count)
                    increment_dist = 20.0
                    await robot.drive_straight(cozmo.util.distance_mm(increment_dist),
                                               cozmo.util.speed_mmps(150)).wait_for_completed()

                    # posX, posY, angle, m_confident = pf.update(odometry, markers_poses)
                    # gui.show_particles(pf.particles)
                    # gui.show_mean(posX,posY,angle,m_confident)
                    # gui.updated.set()

                    count += increment_dist

                if count >= dist:
                    
                    await robot.turn_in_place(cozmo.util.degrees(-angle_rotate_degrees + deliver_wait[2])).wait_for_completed()
                    await robot.set_lift_height(height = 0.0).wait_for_completed()
                    print("Cube Delivered")
                    state = "GLP"


        while (state == "Storage"):
            await robot.wait_for_all_actions_completed()

            if not gone_to_pickup:
                await robot.wait_for_all_actions_completed()
                robot.stop_all_motors()

                newX = storage_wait[0] - posX
                newY = storage_wait[1] - posY
                dist = grid_distance(posX, posY, storage_wait[0], storage_wait[1]) * 25.6
                angle_rotate = atan2(newY, newX)
                angle_rotate_degrees = degrees(angle_rotate)
                diffheading = diff_heading_deg(angle_rotate_degrees, angle)
                #print(angle, otherAngleDegrees, diffheading)
                await robot.turn_in_place(cozmo.util.degrees(diffheading)).wait_for_completed()

                count = 0
                # while (count < dist):
                #     print("Dist Left : ", dist - count)
                #     increment_dist = 20.0
                #     await robot.drive_straight(cozmo.util.distance_mm(increment_dist),
                #                                cozmo.util.speed_mmps(150)).wait_for_completed()
                #     count += increment_dist

                drive_time = 3.0
                speed = dist / drive_time
                await robot.drive_wheels(speed, speed, duration=drive_time)

                
                gone_to_pickup = True
                deliver_wait = storage_deliver[stored_package_count]
                stored_package_count = (stored_package_count + 1) if (stored_package_count > 2) else 0
                state = "Pickup Cube"
                await robot.turn_in_place(cozmo.util.degrees(-angle_rotate_degrees + storage_wait[2])).wait_for_completed()
                print("Goal Reached")

            
            print("State : ", state)
            print("Goal Reached : ", GoneToPose)






    


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
