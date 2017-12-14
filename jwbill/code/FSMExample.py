#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import asyncio
import sys
import cv2
import numpy as np
import cozmo
import time
import os

from cozmo.util import degrees, distance_mm, Speed, radians

from glob import glob

from ExampleTasks.Both import bothCube
from ExampleTasks.Red import redCube
from ExampleTasks.Yellow import yellowCube
from ExampleTasks.findCube import findCube
from find_cube import *
from statemachine import StateMachine

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')
def nothing(x):
    pass

#YELLOW_LOWER = np.array([9, 115, 151])
#YELLOW_UPPER = np.array([179, 215, 255])
#YELLOW_LOWER = np.array([8, 115, 85])
#YELLOW_UPPER = np.array([25, 240, 255])


# Define a decorator as a subclass of Annotator; displays the keypoint
class BoxAnnotator(cozmo.annotate.Annotator):

    cube = None

    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)

        if BoxAnnotator.cube is not None:

            #double size of bounding box to match size of rendered image
            BoxAnnotator.cube = np.multiply(BoxAnnotator.cube,2)

            #define and display bounding box with params:
            #msg.img_topLeft_x, msg.img_topLeft_y, msg.img_width, msg.img_height
            box = cozmo.util.ImageBox(BoxAnnotator.cube[0]-BoxAnnotator.cube[2]/2,
                                      BoxAnnotator.cube[1]-BoxAnnotator.cube[2]/2,
                                      BoxAnnotator.cube[2], BoxAnnotator.cube[2])
            cozmo.annotate.add_img_box_to_image(image, box, "green", text=None)

            BoxAnnotator.cube = None



def run(robot: cozmo.robot.Robot):

    robot.world.image_annotator.annotation_enabled = False
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    fixed_gain,exposure,mode = 3.90,67,1
    FSM = StateMachine()
    find = findCube()
    red = redCube()
    yellow = yellowCube()
    both = bothCube()
    FSM.addState(red)
    FSM.addState(yellow)
    FSM.addState(both)
    FSM.addState(find)
    FSM.setStartState(find)
    try:
        robot.set_head_angle(degrees(-5)).wait_for_completed()
        FSM.run(robot)

#         while True:
#             event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)   #get camera image
#             if event.image is not None:
#                 image = np.asarray(event.image)#cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
#
#
#                 if mode == 1:
#                     robot.camera.enable_auto_exposure = True
#                 else:
#                     robot.camera.set_manual_exposure(exposure,fixed_gain)
#
#                 #find the cube
#                 cube = find_cube(image, YELLOW_LOWER, YELLOW_UPPER)
#                 print(cube)
#                 i = 0
#                 while not cube and i < 5:
#                     cube = find_cube(image, YELLOW_LOWER, YELLOW_UPPER)
# #                    print(cube)
#                     BoxAnnotator.cube = cube
#                     i+=1
#
#                 ################################################################
#
#                 # Todo: Add Motion Here
#                 ################################################################
#
#                 if cube and cube[2] >= 70:
#                    print("stop")
#                 elif cube and cube[0] < 120:
#                     action = robot.turn_in_place(radians(0.1))
#                     await action.wait_for_completed()
#                 elif cube and cube[0] > 180:
#                     action = robot.turn_in_place(radians(-0.1))
#                     await action.wait_for_completed()
#                 elif cube and cube[2] < 70:
#                     action = robot.drive_straight(distance_mm(30), Speed(1000), should_play_anim=False)
#                     await action.wait_for_completed()
#                 else:
#                     action = robot.turn_in_place(radians(0.3))
#                     await action.wait_for_completed()



    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True)
