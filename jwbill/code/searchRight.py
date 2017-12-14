import asyncio
import cozmo
from cozmo.util import degrees, time
from find_cube import find_cube
import numpy as np

from imageShow import showImage

YELLOW_LOWER = np.array([68, 113, 158])
YELLOW_UPPER = np.array([113, 195, 255])

class searchRight:
    def getName(self):
        return "searchRight"

    def run(self, robot: cozmo.robot.Robot):
        robot.set_head_angle(degrees(-15)).wait_for_completed()
        robot.move_lift(-5)
        goLeft = False

        showImage(robot,"RIGHT.PNG")

        while True:
            event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)
            if event.image is not None:
                image = np.asarray(event.image)

                cube = None

                try:
                    cube = find_cube(image, YELLOW_LOWER, YELLOW_UPPER)
                    print(cube)
                    # time.sleep(1)
                except asyncio.TimeoutError:
                    print("Cube not found")

                if cube:
                    return "colorTrack", robot
                else:
                    robot.drive_wheels(25,-25)
                    time.sleep(1)
                    robot.stop_all_motors()