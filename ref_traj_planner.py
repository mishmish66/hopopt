# Do not look in this file the code is so bad

from physics import foot_pos_func, A_func, b_func, Q_func
import numpy as np


def make_single_hop_reference_trajectory(
    q_init,
    qd_init,
    hop_length,
    air_time,
    params,
    contact_height=1.0,
    bottom_height=0.6,
    dt=0.005,
):
    # Unpack params
    m_b, I, m_l, g = params

    cur_theta = q_init[0] + np.pi
    cur_omega = qd_init[0]
    cur_pos = np.array([q_init[1], q_init[2]])
    cur_vel = np.array([qd_init[1], qd_init[2]])

    launch_hvel = hop_length / air_time

    # let's calculate our approximate landing position
    foot_space_height = contact_height
    cur_hpos = cur_pos[0]
    cur_vpos = cur_pos[1]
    cur_hvel = cur_vel[0]
    cur_vvel = cur_vel[1]

    landing_time = 0.0
    landing = cur_vpos > foot_space_height
    decelerating = cur_vvel < 0.0 or landing

    landing_hvel = cur_hvel

    if landing:
        # Now let's compute our landing time (when we are foot_space_height above the ground)
        first_part = -cur_vvel
        second_part = np.sqrt(
            cur_vvel**2 - 4 * (g / 2) * (foot_space_height - cur_vpos)
        )
        landing_time = (first_part - second_part) / (-g)

        target_theta_time = landing_time
        # Let's check if we are above 1.75 foot_space_height
        if landing_time > 0.1:
            # If we are, let's try to be at the landing angle by the pre-landing time instead
            # Hopefully this will give us better angular velocity when we land
            target_theta_time = landing_time - 0.05

        # Now let's compute our landing angle
        landing_hvel = cur_hvel
        landing_vvel = cur_vvel - g * landing_time
        landing_vel = np.array([landing_hvel, landing_vvel])
        landing_angle = np.arctan2(landing_vvel, landing_hvel) + np.pi

        # Let's compute the angular velocity we need to reach the landing angle by the target time
        omega_des = (landing_angle - cur_theta) / target_theta_time

        # Landing com pos
        landing_com_pos = np.array(
            [
                cur_hpos + cur_hvel * landing_time,
                foot_space_height,
            ]
        )
    else:
        contact = True
        landing_vvel = cur_vvel
        landing_vel = cur_vel
        landing_angle = cur_theta
        landing_time = 0.0
        landing_com_pos = np.array([cur_hpos, cur_vpos])

    # Now let's compute our launch angle
    launch_vvel = g * air_time / 2  # + foot_space_height
    launch_vel = np.array([launch_hvel, launch_vvel])
    launch_angle = np.arctan2(launch_vvel, launch_hvel)

    contact_vdec = landing_vvel**2 / 2 / (contact_height - bottom_height)
    dec_time = (
        0.0
        if landing_vvel == 0.0
        else 2 * (bottom_height - contact_height) / landing_vvel
    )
    if cur_vpos < contact_height:
        if cur_vpos > bottom_height:
            min_vdec = 2.5
            contact_vdec = landing_vvel**2 / 2 / (cur_vpos - bottom_height)
            contact_vdec = np.clip(contact_vdec, min_vdec, np.inf)
            dec_time = -landing_vvel / contact_vdec
        else:
            max_decc = 10.0
            dec_time = np.abs(cur_vvel / max_decc)
            contact_vdec = max_decc

    mid_stance_com_vvel = landing_vvel + contact_vdec * dec_time
    mid_stance_com_vpos = (
        landing_com_pos[1]
        + landing_vvel * dec_time
        + 0.5 * contact_vdec * dec_time**2
    )

    if not decelerating:
        bottom_height = cur_vpos
        dec_time = 0.0
        mid_stance_com_vvel = cur_vvel
        mid_stance_com_vpos = cur_vpos

    acc_time = (
        2 * (contact_height - mid_stance_com_vpos) / (launch_vvel + mid_stance_com_vvel)
    )
    contact_vacc = (
        (launch_vvel**2 + mid_stance_com_vvel**2)
        / 2
        / (contact_height - mid_stance_com_vpos)
    )

    contact_hacc = (launch_hvel - landing_hvel) / (dec_time + acc_time)
    mid_stance_com_hpos = (
        landing_com_pos[0]
        + landing_hvel * dec_time
        + 0.5 * contact_hacc * dec_time**2
    )
    contact_dec = np.array([contact_hacc, contact_vdec])
    contact_acc = np.array([contact_hacc, contact_vacc])
    mid_stance_com_pos = np.array([mid_stance_com_hpos, mid_stance_com_vpos])

    mid_stance_com_vel = landing_vel + contact_dec * dec_time

    # Now let's compute the launching com pos

    launch_com_pos = (
        mid_stance_com_pos
        + mid_stance_com_vel * acc_time
        + 0.5 * contact_acc * acc_time**2
    )

    # Now let's actually build the trajectory
    interp_time_step = dt
    interp_steps = int(
        (dec_time + acc_time + air_time + landing_time) / interp_time_step
    )
    interp_step_times = np.arange(interp_steps) * interp_time_step

    traj = np.zeros((interp_steps, 6))

    pre_contact_cutoff = int(landing_time / interp_time_step)

    pre_contact_times = interp_step_times[:pre_contact_cutoff]

    if landing:
        # Pre-contact flight phase
        # Set the angular position before contact
        traj[:pre_contact_cutoff, 0] = (
            np.where(
                pre_contact_times < target_theta_time,
                cur_theta + omega_des * pre_contact_times,
                landing_angle,
            )
            - np.pi
        )
        # Set the angular velocity before contact
        traj[:pre_contact_cutoff, 3] = np.where(
            pre_contact_times < target_theta_time,
            omega_des,
            0,
        )
        # Set the horizontal position before contact
        traj[:pre_contact_cutoff, 1] = cur_hpos + cur_hvel * pre_contact_times
        # Set the horizontal velocity before contact
        traj[:pre_contact_cutoff, 4] = cur_hvel
        # Set the vertical position before contact
        traj[:pre_contact_cutoff, 2] = (
            cur_vpos + cur_vvel * pre_contact_times - 0.5 * g * pre_contact_times**2
        )
        # Set the vertical velocity before contact
        traj[:pre_contact_cutoff, 5] = cur_vvel + g * pre_contact_times

    # Contact deceleration phase
    contact_acc_time = landing_time + dec_time
    contact_dec_cutoff = int(contact_acc_time / interp_time_step)
    contact_cutoff = int((contact_acc_time + acc_time) / interp_time_step)
    contact_dec_times = interp_step_times[pre_contact_cutoff:contact_dec_cutoff]

    contact_dec_vels = (
        landing_vel + contact_dec * (contact_dec_times - landing_time)[:, None]
    )
    # Set the contact dec xy velocities
    traj[pre_contact_cutoff:contact_dec_cutoff, 4:6] = contact_dec_vels

    # set the contact dec xy positions
    stance_dec_com_xy_pos = (
        landing_com_pos
        + landing_vel * (contact_dec_times - landing_time)[:, None]
        + 0.5 * contact_dec * (contact_dec_times - landing_time)[:, None] ** 2
    )
    traj[pre_contact_cutoff:contact_dec_cutoff, 1:3] = stance_dec_com_xy_pos

    # Contact acceleration phase
    contact_acc_times = interp_step_times[contact_dec_cutoff:contact_cutoff]
    contact_acc_vels = (
        mid_stance_com_vel
        + contact_acc * (contact_acc_times - contact_acc_time)[:, None]
    )
    contact_acc_com_xy_pos = (
        mid_stance_com_pos
        + mid_stance_com_vel * (contact_acc_times - contact_acc_time)[:, None]
        + 0.5 * contact_acc * (contact_acc_times - contact_acc_time)[:, None] ** 2
    )

    # set the contact acc xy velocities
    traj[contact_dec_cutoff:contact_cutoff, 4:6] = contact_acc_vels
    # set the contact acc xy positions
    traj[contact_dec_cutoff:contact_cutoff, 1:3] = contact_acc_com_xy_pos

    # Concat the stance phase
    stance_com_xy_pos = np.concatenate(
        [stance_dec_com_xy_pos, contact_acc_com_xy_pos], axis=0
    )
    contact_vels = np.concatenate([contact_dec_vels, contact_acc_vels], axis=0)

    launch_time = contact_acc_time + acc_time

    # set the contact theta
    # calculate contact foot pos
    q_land = np.array([landing_angle + np.pi, *landing_com_pos, 0])
    contact_point_pos = foot_pos_func(q_land)
    # Calculate relative foot pos over stance phase
    contact_point_rel_pos = stance_com_xy_pos - contact_point_pos.T
    # Calculate angles
    contact_point_angles = np.arctan2(
        contact_point_rel_pos[:, 1], contact_point_rel_pos[:, 0]
    )
    traj[pre_contact_cutoff:contact_cutoff, 0] = contact_point_angles - np.pi

    # set the contact theta dot
    (stance_theta_dot,) = (np.cross(contact_point_rel_pos, contact_vels, axis=-1),)
    traj[pre_contact_cutoff:contact_cutoff, 3] = stance_theta_dot

    # Flight phase
    flight_times = interp_step_times[contact_cutoff:]

    # Flight angular velocity
    flight_end_theta = -launch_angle + np.pi
    if len(contact_point_angles) > 0:
        flight_start_theta = contact_point_angles[-1]
    else:
        flight_start_theta = cur_theta

    flight_omega = (flight_end_theta - flight_start_theta) / air_time
    flight_acc = np.array([0, -g])

    # set the flight theta
    traj[contact_cutoff:, 0] = (
        flight_start_theta + flight_omega * (flight_times - launch_time) - np.pi
    )

    # set the flight theta dot
    traj[contact_cutoff:, 3] = flight_omega

    if contact_cutoff > 0:
        # set the flight xy pos
        traj[contact_cutoff:, 1:3] = (
            traj[contact_cutoff - 1, 1:3]
            + launch_vel * (flight_times - launch_time)[:, None]
            + 0.5 * flight_acc * (flight_times - launch_time)[:, None] ** 2
        )
        # set the flight xy vel
        traj[contact_cutoff:, 4:6] = (
            launch_vel + flight_acc * (flight_times - launch_time)[:, None]
        )
    else:
        traj[contact_cutoff:, 1:3] = (
            cur_pos
            + launch_vel * (flight_times - launch_time)[:, None]
            + 0.5 * flight_acc * (flight_times - launch_time)[:, None] ** 2
        )
        # set the flight xy vel
        traj[contact_cutoff:, 4:6] = (
            cur_vel + flight_acc * (flight_times - launch_time)[:, None]
        )

    contact = np.zeros((interp_steps, 1), dtype=bool)
    contact[pre_contact_cutoff:contact_cutoff] = True
    return traj, contact
