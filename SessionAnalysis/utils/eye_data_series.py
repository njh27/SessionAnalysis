import numpy as np



def mode1D(x):
    # Faster and without scipy way to find mode of 1D array
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def find_saccade_windows(x_vel, y_vel, ind_cushion=20, acceleration_thresh=1, speed_thresh=30):
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
    threshold_indices = np.where(np.logical_or((eye_speed > speed_thresh), (np.absolute(eye_acceleration) > acceleration_thresh)))[0]
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


def find_eye_offsets(x_pos, y_pos, x_vel, y_vel, epsilon_eye=0.1, max_iter=10,
                    ind_cushion=20, acceleration_thresh=1, speed_thresh=30):
    """ Find the DC offsets in eye position and velocity during the window
    input in the position and velocity data. Done by taking the mode of their
    values, then finding saccades using 'find_saccade_windows', then repeating
    iteratively until convergence within epsilon_eye improveming over iterations
    or max_iter iterations is reached. The saccade removal threshold implies
    that the mode value of eye velocity during the input data needs to be less
    than speed_thresh for this to work. The script attempts to make this true
    but it should be noted that a sufficient sample is needed for success, such
    that there are enough saccade-free data samples to get a good estimate. To
    this end, it probably makes the most sense that the mode eye velocity across
    all (0, 0) fixation epochs is first subtracted.
    Data inputs are assumed to be 1D numpy arrays.
    Output: offsets = [x_pos_offset, y_pos_offset, x_vel_offset, y_vel_offset]
    """
    # Need to copy data so not overwritten in the inputs
    x_pos, y_pos = np.copy(x_pos), np.copy(y_pos)
    x_vel, y_vel = np.copy(x_vel), np.copy(y_vel)
    offsets = [0, 0, 0, 0]
    # Everything will be found as a saccade unless this is satisfied
    eye_speed = np.sqrt(x_vel**2 + y_vel**2)
    n_iters = 0
    while np.all(eye_speed >= speed_thresh):
        x_vel_median = np.median(x_vel)
        y_vel_median = np.median(y_vel)
        x_vel = x_vel - x_vel_median
        y_vel = y_vel - y_vel_median
        eye_speed = np.sqrt(x_vel**2 + y_vel**2)
        n_iters += 1
        offsets[2] += x_vel_median
        offsets[3] += y_vel_median
        if n_iters >= max_iter:
            raise ValueError("Input eye velocity data's offset is much greater than speed threshold and could not be fixed")

    # Now find mode velocities and saccades and iteratively reduce
    delta_eye = 0
    n_iters = 0
    # Start assuming no saccades
    saccade_index = np.zeros(x_vel.shape[0], dtype='bool')
    while n_iters < max_iter:
        # Update velocity
        x_vel_mode, _ = mode1D(x_vel[~saccade_index])
        x_vel = x_vel - x_vel_mode
        offsets[2] += x_vel_mode
        y_vel_mode, _ = mode1D(y_vel[~saccade_index])
        y_vel = y_vel - y_vel_mode
        offsets[3] = y_vel_mode

        # Update position
        x_pos_mode, _ = mode1D(x_pos[~saccade_index])
        x_pos = x_pos - x_pos_mode
        offsets[0] += x_pos_mode
        y_pos_mode, _ = mode1D(y_pos[~saccade_index])
        y_pos = y_pos - y_pos_mode
        offsets[1] = y_pos_mode

        # Current modes are the magnitude of the delta offset update
        delta_eye = np.amax(np.abs((x_pos_mode, y_pos_mode, x_vel_mode, y_vel_mode)))
        if delta_eye < epsilon_eye:
            # This trial's offsets are good enough so we are done
            break
        else:
            # Try again to reduce offsets. Update saccade windows/index
            _, saccade_index = find_saccade_windows(x_vel, y_vel,
                                ind_cushion=ind_cushion,
                                acceleration_thresh=acceleration_thresh,
                                speed_thresh=speed_thresh)
            n_iters += 1

    # offsets = [x_pos_offset, y_pos_offset, x_vel_offset, y_vel_offset]
    return offsets
