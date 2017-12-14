# -----------------------------------
# --- Naud Ghebre and Arihan Shah ---
# -----------------------------------
from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np


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

    x_prev = odom[0]
    x = odom[1]

    # del_rot1 = proj_angle_deg(math.degrees(math.atan2(x[1] - x_prev[1], x[0] - x_prev[0]))
    #                           - proj_angle_deg(x_prev[2]))
    # del_trans = math.sqrt((x[0] - x_prev[0])**2 + (x[1] - x_prev[1])**2)
    # del_rot2 = proj_angle_deg(x[2]) - proj_angle_deg(x_prev[2]) - proj_angle_deg(del_rot1)


    # # del_rot1_hat = del_rot1 - proj_angle_deg(random.gauss(0.0, alpha1 * del_rot1 + alpha2 * del_trans))
    # # del_trans_hat = del_trans - random.gauss(0.0, alpha3 * del_trans + alpha4 * (del_rot1 + del_rot2))
    # # del_rot2_hat = del_rot2 - proj_angle_deg(random.gauss(0.0, alpha1 * del_rot2 + alpha2 * del_trans))

    # del_rot1_hat = del_rot1 - proj_angle_deg(random.gauss(0.0, alpha1 * del_rot1 + alpha2 * del_trans))
    # del_trans_hat = del_trans - random.gauss(0.0, alpha3 * del_trans + alpha4 * (del_rot1 + del_rot2))
    # del_rot2_hat = del_rot2 - proj_angle_deg(random.gauss(0.0, alpha1 * del_rot2 + alpha2 * del_trans))

    newParticles = []

    for particle in particles:
        del_rot1 = proj_angle_deg(math.degrees(math.atan2(x[1] - x_prev[1], x[0] - x_prev[0]))) - proj_angle_deg(
            x_prev[2])
        del_trans = math.sqrt((x[0] - x_prev[0]) ** 2 + (x[1] - x_prev[1]) ** 2)
        del_rot2 = proj_angle_deg(x[2]) - proj_angle_deg(x_prev[2]) - proj_angle_deg(del_rot1)

        # del_rot1_hat = del_rot1 - proj_angle_deg(random.gauss(0.0, alpha1 * del_rot1 + alpha2 * del_trans))
        # del_trans_hat = del_trans - random.gauss(0.0, alpha3 * del_trans + alpha4 * (del_rot1 + del_rot2))
        # del_rot2_hat = del_rot2 - proj_angle_deg(random.gauss(0.0, alpha1 * del_rot2 + alpha2 * del_trans))

        del_rot1_hat = add_gaussian_noise(del_rot1, ODOM_HEAD_SIGMA)
        del_trans_hat = add_gaussian_noise(del_trans, ODOM_TRANS_SIGMA)
        del_rot2_hat = add_gaussian_noise(del_rot2, ODOM_HEAD_SIGMA)

        # h = particle.h + del_rot1_hat
        # dx = math.cos(math.radians(h)) * del_trans_hat
        # dy = math.sin(math.radians(h)) * del_trans_hat
        # p = Particle(particle.x + dx, particle.y + dy, h + del_rot2_hat)
        particle.move(del_rot1_hat, del_trans_hat, del_rot2_hat)
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

    # if len(measured_marker_list) == 0:
    #     return particles

    weights = []

    for particle in particles:

        markers_visible_to_particle = particle.read_markers(grid)
        markers_measured = measured_marker_list.copy()

        if not grid.is_in(particle.x, particle.y) or not grid.is_free(particle.x, particle.y):
            weights.append(0)
        elif len(markers_measured) == 0:
            weights.append(1 / len(particles))
        elif len(markers_visible_to_particle) == 0:
            weights.append(0)
        elif len(markers_measured) > len(markers_visible_to_particle):
            weights.append(0)
        else:
            pairings = []
            while markers_measured:

                dist = [(m1, m2, grid_distance(m1[0], m1[1], m2[0], m2[1])) for m1 in markers_measured
                        for m2 in markers_visible_to_particle]

                if dist:
                    pair = min(dist, key=lambda x: x[2])[0:2]
                    pairings.append(pair)
                    markers_visible_to_particle.remove(pair[1])
                    markers_measured.remove(pair[0])

            prob = 1.0

            for landmark in pairings:
                d = grid_distance(landmark[0][0], landmark[0][1], landmark[1][0], landmark[1][1])
                a = diff_heading_deg(landmark[0][2], landmark[1][2])
                prob *= math.exp(- ((d / MARKER_TRANS_SIGMA) ** 2 + (a / MARKER_ROT_SIGMA) ** 2) / 2)
            weights.append(prob)

    measured_particles = []
    sum_weight = sum(weights)
    weights_normalized = []

    if (sum_weight != 0):
        weights_normalized = [x / sum_weight for x in weights]
    else:
        for i in range(len(particles)):
            particles[i] = Particle(*grid.random_free_place())
        weights_normalized = [1 / (len(particles))] * len(particles)

    index = np.random.choice(a=range(len(particles)), size=len(weights), p=weights_normalized)

    for i in index:
        if weights_normalized[i] < (0.003 * (1 / len(particles))):
            measured_particles.append(Particle(*grid.random_free_place()))
        else:
            measured_particles.append(Particle(particles[i].x, particles[i].y, particles[i].h))

    return measured_particles

    # if not sum_weight <= .05 / len(particles):
    #     weights_normalized = [x / sum_weight for x in weights]
    #     measured_particles = np.random.choice(a=particles, size=len(particles), replace=True, p=weights_normalized)
    # else:
    #     # count = 0
    #     # for weight in weights:
    #     #     if weight == 0:
    #     #         count += 1
    #     #
    #     # if count != len(particles):
    #     #     avg = sum_weight / (len(particles) - count)
    #     # else:
    #     #     avg = 0

    #     # measured_particles = np.random.choice(a=particles, size=len(particles), replace=True, p=weights_normalized)

    #     for i in range(len(particles)):
    #         measured_particles.append(Particle.create_random(1, grid)[0])

    return measured_particles


# from grid import *
# from particle import Particle
# from utils import *
# from setting import *
#
# # ------------------------------------------------------------------------
# def motion_update(particles, odom):
#     """ Particle filter motion update
#
#         Arguments:
#         particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
#                 before motion update
#         odom -- noisy odometry measurement, a pair of robot pose, i.e. last time
#                 step pose and current time step pose
#
#         Returns: the list of particle represents belief \tilde{p}(x_{t} | u_{t})
#                 after motion update
#     """
#     return particles
#
# # ------------------------------------------------------------------------
# def measurement_update(particles, measured_marker_list, grid):
#     """ Particle filter measurement update
#
#         Arguments:
#         particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
#                 before meansurement update
#         measured_marker_list -- robot detected marker list, each marker has format:
#                 measured_marker_list[i] = (rx, ry, rh)
#                 rx -- marker's relative X coordinate in robot's frame
#                 ry -- marker's relative Y coordinate in robot's frame
#                 rh -- marker's relative heading in robot's frame, in degree
#         grid -- grid world map, which contains the marker information,
#                 see grid.h and CozGrid for definition
#
#         Returns: the list of particle represents belief p(x_{t} | u_{t})
#                 after measurement update
#     """
#     measured_particles = []
#     return measured_particles
