# Samuel Conrad and Josh Jibilian

from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy


# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurement, a pair of robot pose, i.e. last time
                step pose and current time step pose

        Returns: the list of particle represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    newParticles = []

    oldPos = odom[0]
    newPos = odom[1]
    alpha1 = 0.001
    alpha2 = 0.001
    alpha3 = 0.001
    alpha4 = 0.001

    rot1 = math.degrees(math.atan2(newPos[1] - oldPos[1], newPos[0] - oldPos[0])) - oldPos[2]
    trans = grid_distance(oldPos[0], oldPos[1], newPos[0], newPos[1])
    rot2 = newPos[2] - oldPos[2] - rot1

    newRot1 = rot1 - add_gaussian_noise(alpha1 * rot1 + alpha2 * trans, ODOM_HEAD_SIGMA)
    newTrans = trans - add_gaussian_noise(alpha3 * trans + alpha4 * (rot1 + rot2), ODOM_TRANS_SIGMA)
    newRot2 = rot2 - add_gaussian_noise(alpha1 * rot2 + alpha2 * trans, ODOM_HEAD_SIGMA)

    for particle in particles:
        particle.x = particle.x + newTrans * math.cos(math.radians(particle.h + newRot1))
        particle.x = add_gaussian_noise(particle.x, ODOM_TRANS_SIGMA)

        particle.y = particle.y + newTrans * math.sin(math.radians(particle.h + newRot1))
        particle.y = add_gaussian_noise(particle.y, ODOM_TRANS_SIGMA)

        particle.h = particle.h + newRot1 + newRot2
        particle.h = add_gaussian_noise(particle.h, ODOM_HEAD_SIGMA)

        newParticles.append(particle)

    return newParticles


# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map, which contains the marker information, 
                see grid.h and CozGrid for definition

        Returns: the list of particle represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    counter = 0
    weightArr = []

    if len(measured_marker_list) != 0:
        for particle in particles:
            # Obtain list of localization markers 
            visibleMarkers = particle.read_markers(grid)

            shouldAppendParticle = particle.x >= grid.width or particle.x < 0 or particle.y >= grid.height or particle.y < 0 or (
                                                                                                                                particle.x,
                                                                                                                                particle.y) in grid.occupied
            if shouldAppendParticle:
                weightArr.append((particle, 0))
            else:
                mmlLength = len(measured_marker_list)
                vmLength = len(visibleMarkers)
                pairs = []
                for measuredMarker in measured_marker_list:
                    if len(visibleMarkers) != 0:
                        # find closest marker
                        nearestMarker = findNearestMarker(measuredMarker, visibleMarkers)
                        # remove from possible future pairings
                        visibleMarkers.remove(nearestMarker)
                        # store pairings
                        pairs.append((nearestMarker, measuredMarker))

                weightArr.append((particle, getProbability(pairs, mmlLength, vmLength)))

        counter2 = 0
        remove = int(PARTICLE_COUNT / 100)

        # update weights
        weightArr.sort(key=lambda x: x[1])
        weightArr = weightArr[remove:]
        for i, j in weightArr:
            if j != 0:
                counter2 = counter2 + j
            if j == 0:
                counter = counter + 1
        weightArr = weightArr[counter:]
        counter = counter + remove
    else:
        counter2 = 1
        for p in particles:
            weightArr.append((p, 1 / len(particles)))

    particleList = []
    weightList = []
    for i, j in weightArr:
        weight = j / counter2
        weightList.append(weight)
        particleList.append(Particle(i.x, i.y, i.h))

    # Create new particle list by random
    newParticleList = []
    if particleList != []:
        newParticleList = numpy.random.choice(particleList, size=len(particleList), replace=True, p=weightList)

    measured_particles = getMeasuredParticles(Particle.create_random(counter, grid)[:], newParticleList)

    return measured_particles


def findNearestMarker(measuredMarker, visibleMarkers):
    measuredMarkerX, measuredMarkerY, _ = add_marker_measurement_noise(measuredMarker, MARKER_TRANS_SIGMA,
                                                                       MARKER_ROT_SIGMA)
    nearestMarker = visibleMarkers[0]
    nearestDistance = grid_distance(measuredMarkerX, measuredMarkerY, nearestMarker[0], nearestMarker[1])
    for visibleMarker in visibleMarkers:
        visibleMarkerX = visibleMarker[0]
        visibleMarkerY = visibleMarker[1]
        distance = grid_distance(measuredMarkerX, measuredMarkerY, visibleMarkerX, visibleMarkerY)
        if distance < nearestDistance:
            nearestMarker = visibleMarker
            nearestDistance = distance

    return nearestMarker


def getProbability(pairs, mmlLength, vmLength):
    probability = 1
    transConstantMax = 0
    for p1, p2 in pairs:
        # euclidean distance of each pair
        markerDistance = grid_distance(p1[0], p1[1], p2[0], p2[1])
        markerAngle = diff_heading_deg(p1[2], p2[2])

        # a: distBetweenMarkers^2 / (2 * (standard deviation of the gaussian model for translation)^2)
        transConstantMax = max(transConstantMax, (markerDistance ** 2) / (2 * (MARKER_TRANS_SIGMA ** 2)))

        # b: angleBetweenMarkers^2 / (2 * (standard deviation for rotation measurements)^2)
        newRotConstant = (markerAngle ** 2) / (2 * (MARKER_ROT_SIGMA ** 2))

        # p = e^-(a + b)
        power = ((markerDistance ** 2) / (2 * (MARKER_TRANS_SIGMA ** 2))) + newRotConstant
        probability = probability * numpy.exp(-power)

    # handle measured marker and visible marker difference
    rotConstantMax = (45 ** 2) / (2 * (MARKER_ROT_SIGMA ** 2))
    difference = math.fabs(mmlLength - vmLength)
    count = 0
    while count < int(difference):
        probability = probability * numpy.exp(-transConstantMax - rotConstantMax)
        count = count + 1

    return probability


def getMeasuredParticles(measured_particles, newParticleList):
    for particle in newParticleList:
        particleX = add_gaussian_noise(particle.x, ODOM_TRANS_SIGMA)
        particleY = add_gaussian_noise(particle.y, ODOM_TRANS_SIGMA)
        particleH = add_gaussian_noise(particle.h, ODOM_HEAD_SIGMA)
        newParticle = Particle(particleX, particleY, particleH)
        measured_particles.append(newParticle)

    return measured_particles