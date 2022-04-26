import numpy as np
from scipy import stats



def find_saccade_windows(x_vel, y_vel, ind_cushion=20, acceleration_thresh=1, velocity_thresh=30):
    """ Given the x and y velocity timeseries and threshold criteria, this will
    find the time window indices from saccade start to stop +/- time cushion.
    x_vel and y_vel are assumed to be 1D numpy arrays.
    """
    # Force ind_cushion to be integer so saccade windows are integers that can be used as indices
    ind_cushion = np.array(ind_cushion).astype('int')

    # Make this just for 1 trial!
    # Compute normalized eye speed vector and corresponding acceleration
    # print(trial)
    # print(maestro_PL2_data[trial]['horizontal_eye_velocity'])
    eye_speed = np.sqrt(x_vel**2 + y_vel**2)
    eye_acceleration = np.zeros(eye_speed.shape[0])
    eye_acceleration[1:] = np.diff(eye_speed, n=1, axis=0)

    # Find saccades as all points exceeding input thresholds
    threshold_indices = np.where(np.logical_or((eye_speed > velocity_thresh), (np.absolute(eye_acceleration) > acceleration_thresh)))[0]
    if threshold_indices.size == 0:
        # No saccades this trial
        saccade_windows = np.empty(0, dtype=np.int64)
        saccade_index = np.zeros(eye_speed.shape[0], 'bool')
        return saccade_windows, saccade_index

    # Find the end of all saccades based on the time gap between each element of threshold indices.
    # Gaps between saccades must exceed the ind_cushion padding on either end to count as separate saccades.
    switch_indices = np.zeros_like(threshold_indices, dtype='bool')
    # Needs to have a difference of at least 1
    switch_indices[0:-1] = np.diff(threshold_indices) > max(ind_cushion * 2, 1)

    # Use switch index to find where one saccade ends and the next begins and store the locations.  These don't include
    # the beginning of first saccade or end of last saccade, so add 1 element to saccade_windows
    switch_points = np.where(switch_indices)[0]
    saccade_windows = np.full((switch_points.shape[0] + 1, 2), np.nan, dtype=np.int64)

    # Check switch points and use their corresponding indices to find actual time values in threshold_indices
    for saccade in range(0, saccade_windows.shape[0]):
        if saccade == 0:
            # Start of first saccade not indicated in switch points so add time cushion to the first saccade
            # time in threshold indices after checking whether it is within time limit of trial data
            if threshold_indices[saccade] - ind_cushion > 0:
                saccade_windows[saccade, 0] = threshold_indices[saccade] - ind_cushion
            else:
                saccade_windows[saccade, 0] = 0

        # Making this an "if" rather than "elif" means in case of only 1 saccade, this statement
        # and the "if" statement above both run
        if saccade == saccade_windows.shape[0] - 1:
            # End of last saccade, which is not indicated in switch points so add time cushion to end of
            # last saccade time in threshold indices after checking whether it is within time of trial data
            if eye_speed.shape[0] < threshold_indices[-1] + ind_cushion:
                saccade_windows[saccade, 1] = eye_speed.shape[0]
            else:
                saccade_windows[saccade, 1] = threshold_indices[-1] + ind_cushion
        else:
            # Add cushion to end of current saccade.
            saccade_windows[saccade, 1] = threshold_indices[switch_points[saccade]] + ind_cushion

        # Add cushion to the start of next saccade if there is one
        if saccade_windows.shape[0] > saccade + 1:
            saccade_windows[saccade + 1, 0] = threshold_indices[switch_points[saccade] + 1] - ind_cushion

    # Set boolean index for marking saccade times and save
    saccade_index = np.zeros(eye_speed.shape[0], 'bool')
    for saccade_win in saccade_windows:
        saccade_index[saccade_win[0]:saccade_win[1]] = True

    return saccade_windows, saccade_index


def find_eye_offsets(x_eye, y_eye, align_segment=3, fixation_target=0, fixation_window=(-200, 0), epsilon_eye=0.1, max_iter=10,
                         ind_cushion=20, acceleration_thresh=1, velocity_thresh=30):
    """ This subtracts the DC offsets in eye position and velocity by taking the mode of their values in the fixation window aligned on
        align_segment.  After this first adjustment, saccades are found in the fixation window and the mode eye position and velocity
        values are recomputed without saccades and subtracting again.  This is repeated recursively until the absolute value of the
        position and velocity mode during fixation window is less than epsilon_eye or max_iter recursive calls is reached.  The call to
        find_saccade_windows means that this variable will also be assigned to maestro_PL2_data.  The modes
        for both position and velocity are only computed for values where eye_velocity is less than velocity_thresh.  DC offsets in
        eye velocity greater than velocity_thresh will cause this to fail.
        """

    delta_eye = np.zeros(len(maestro_PL2_data))
    n_iters = 0
    next_trial_list = [x for x in range(0, len(maestro_PL2_data))]
    while n_iters < max_iter and len(next_trial_list) > 0:
        trial_indices = next_trial_list
        next_trial_list = []
        for trial in trial_indices:# len(maestro_PL2_data)):
            if not maestro_PL2_data[trial]['maestro_events'][align_segment]:
                # Current trial lacks desired align_segment
                continue

            align_time = maestro_PL2_data[trial]['targets'][fixation_target].get_next_refresh(maestro_PL2_data[trial]['maestro_events'][align_segment][0])
            time_index = np.arange(align_time + fixation_window[0], align_time + fixation_window[1] + 1, 1, dtype='int')

            if 'saccade_windows' in maestro_PL2_data[trial]:
                saccade_index = maestro_PL2_data[trial]['saccade_index'][time_index]
                time_index = time_index[~saccade_index]
                if np.all(saccade_index):
                    # Entire fixation window has been marked as a saccade
                    # TODO This is an error, or needs to be fixed, or skipped??
                    print("Entire fixation window marked as saccade for trial {} and was skipped".format(trial))
                    continue

            if (not np.any(np.abs(maestro_PL2_data[trial]['eye_velocity'][0][time_index]) < velocity_thresh) or
                not np.any(np.abs(maestro_PL2_data[trial]['eye_velocity'][1][time_index]) < velocity_thresh)):
                # Entire fixation window is over velocity threshold
                # TODO This is an error, or needs to be fixed, or skipped??
                print("Entire fixation window marked as saccade for trial {} and was skipped".format(trial))
                continue

            # Get modes of eye data in fixation window, excluding saccades and velocity times over threshold, then subtract them from eye data
            horizontal_vel_mode, n_horizontal_vel_mode = stats.mode(maestro_PL2_data[trial]['eye_velocity'][0][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][0][time_index]) < velocity_thresh])
            maestro_PL2_data[trial]['eye_velocity'][0] = maestro_PL2_data[trial]['eye_velocity'][0] - horizontal_vel_mode
            vertical_vel_mode, n_vertical_vel_mode = stats.mode(maestro_PL2_data[trial]['eye_velocity'][1][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][1][time_index]) < velocity_thresh])
            maestro_PL2_data[trial]['eye_velocity'][1] = maestro_PL2_data[trial]['eye_velocity'][1] - vertical_vel_mode


            # Position mode is also taken when VELOCITY is less than threshold and adjusted by the difference between eye data and target position
            h_position_offset = maestro_PL2_data[trial]['eye_position'][0][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][0][time_index]) < velocity_thresh] - np.nanmean(maestro_PL2_data[trial]['targets'][fixation_target].position[0, time_index])
            horizontal_pos_mode, n_horizontal_pos_mode = stats.mode(h_position_offset)
            maestro_PL2_data[trial]['eye_position'][0] = maestro_PL2_data[trial]['eye_position'][0] - horizontal_pos_mode
            v_position_offset = maestro_PL2_data[trial]['eye_position'][1][time_index][np.abs(maestro_PL2_data[trial]['eye_velocity'][1][time_index]) < velocity_thresh] - np.nanmean(maestro_PL2_data[trial]['targets'][fixation_target].position[1, time_index])
            vertical_pos_mode, n_vertical_pos_mode = stats.mode(v_position_offset)
            maestro_PL2_data[trial]['eye_position'][1] = maestro_PL2_data[trial]['eye_position'][1] - vertical_pos_mode

            delta_eye[trial] = np.amax(np.abs((horizontal_vel_mode, vertical_vel_mode, horizontal_pos_mode, vertical_pos_mode)))
            if delta_eye[trial] >= epsilon_eye:
                # This trial's offsets aren't good enough so try again next loop
                next_trial_list.append(trial)

        maestro_PL2_data = find_saccade_windows(maestro_PL2_data, ind_cushion=ind_cushion, acceleration_thresh=acceleration_thresh, velocity_thresh=velocity_thresh)
        n_iters += 1

    return maestro_PL2_data
