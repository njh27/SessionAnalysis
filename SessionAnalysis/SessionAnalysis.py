import numpy as np
from scipy.stats import mode
from scipy.signal import savgol_filter
from numpy.linalg import norm
import maestroPL2
import sys




from collections.abc import MutableSequence
class ConjoinedList(MutableSequence):
    """ Objects of this class essentially behave as lists for purposes of storing,
        indexing and iterating. Their key feature is they can be conjoined to
        other ConjoinedList objects to enforce both lists to remain in a one-to-one
        index mapping.  If elements from one list are deleted, the same elements
        are deleted from all of its conjoined lists. Functionality that adds
        list elements is not supported. """
    def __init__(self, data_list):
        self._initialized_list = False
        self.__list__ = list(data_list)
        self._conjoined_lists = []
        self._initialized_list = True

    def __delitem__(self, index):
        del self.__list__[index]
        if sys._getframe().f_back.f_code.co_name == '__delitem__':
            # Prevent recursion between conjoined lists
            return
        for con_lst in self._conjoined_lists:
            del con_lst[index]

    def insert(self, index, value):
        if self._initialized_list:
            raise ValueError("Can't insert items for ConjoinedList type")
        else:
            self.__list__.insert(index, value)

    def __setitem__(self, index, value):
        self.__list__[index] = value

    def __getitem__(self, index):
        return self.__list__[index]

    def __len__(self):
        return len(self.__list__)

    def conjoin_list(self, NewConjoinedList):
        if len(NewConjoinedList) != len(self):
            raise ValueError("Additional lists must be of the same length!")
        self._conjoined_lists.append(NewConjoinedList)
        try:
            NewConjoinedList._conjoined_lists.append(self)
        except AttributeError as err:
            raise AttributeError("Input 'NewConjoinedList' must have a"
                "conjoined_lists attribute to conjoin") from err

    def __str__(self):
        return f"{self.__list__}"

    def __repr__(self):
        return f"SessionAnalysis.ConjoinedList({self.__list__})"


class Session(object):
    """ Class encompassing trials and their Maestro data as well as any potential
        plexon data.  Contains trial info and block orders and can be referenced
        by neurons as needed. """

    def __init__(self, maestro_data, name=None, blocks=None, neurons=None, sacc_time_cushion=20,
                 nan_saccades=True, acceleration_thresh=1, velocity_thresh=30):
        if name is None:
            name_start = len(maestro_data[0]['filename']) - maestro_data[0]['filename'][-1::-1].find('/')
            name_stop = len(maestro_data[0]['filename']) - maestro_data[0]['filename'][-1::-1].find('.') - 1
            self.name = maestro_data[0]['filename'][name_start:name_stop]
        else:
            self.name = name
        self.blocks = {} if blocks is None else blocks
        self.neurons = neurons
        self.maestro_data_2_trials(maestro_data)
        self.neurons = []

        # Set values that will be used for any functions that depend on saccade removal
        self.sacc_time_cushion = sacc_time_cushion
        self.nan_saccades = nan_saccades
        self.acceleration_thresh = acceleration_thresh
        self.velocity_thresh = velocity_thresh

    def maestro_data_2_trials(self, maestro_data):
        """ Streamline the Maestro data list into a list of trials """
        self.trials = ConjoinedList([{} for x in range(0, len(maestro_data))])
        for t in range(0, len(maestro_data)):
            self.trials[t]['trial_name'] = maestro_data[t]['trial_name']
            self.trials[t]['set_name'] = maestro_data[t]['set_name']
            self.trials[t]['sub_set_name'] = maestro_data[t]['sub_set_name']
            self.trials[t]['timestamp'] = maestro_data[t]['timestamp']
            self.trials[t]['duration_ms'] = maestro_data[t]['duration_ms']
            self.trials[t]['maestro_events'] = maestro_data[t]['maestro_events']
            self.trials[t]['time_series'] = maestro_data[t]['time_series']
            self.trials[t]['time_series_alignment'] = maestro_data[t]['time_series_alignment']
            self.trials[t]['eye_position'] = maestro_data[t]['eye_position']
            self.trials[t]['eye_velocity'] = maestro_data[t]['eye_velocity']
            self.trials[t]['targets'] = maestro_data[t]['targets']
            self.trials[t]['plexon_start_stop'] = maestro_data[t]['plexon_start_stop']
            self.trials[t]['plexon_events'] = maestro_data[t]['plexon_events']

    def add_neuron(self, Neuron):
        """. """
        Neuron.join_session(self)

    def get_trial_names(self):
        trial_names = []
        for trial in self.trials:
            if trial['trial_name'] not in trial_names:
                trial_names.append(trial['trial_name'])
        return trial_names

    def remove_trials_less_than_event(self, name_event_num_dict):
        """ Remove trials that do not contain the required number of Maestro events
            (e.g. trials that terminated too early to be useful).  The required
            number of events is specified by the input dictionary name_event_num_dict
            that contains all trial names and returns the number of events required
            for that trial name to avoid deletion. """
        for t in range(len(self.trials) - 1, -1, -1):
            if not self.trials[t]['maestro_events'][name_event_num_dict[self.trials[t]['trial_name']]]:
                del self.trials[t]
                for n in range(0, len(self.neurons)):
                    del self.neurons[n].trial_spikes[t]

    def add_block(self, trial_names, block_name, criteria_dict):
        """ Adds groups of trials to blocks named 'block_name'. """
        trial_windows, stop_triggers = maestroPL2.find_trial_blocks(self.trials, trial_names, **criteria_dict)
        if len(trial_windows) > 0:
            # for blk in range(0, len(trial_windows)):
            #     if len(trial_windows) > 1:
            #         block_name = block_name + str(blk)
            self.blocks[block_name] = {}
            self.blocks[block_name]['trial_windows'] = trial_windows
            self.blocks[block_name]['trial_names'] = trial_names

    def trim_block(self, block, remove_block):
        """ Trims overlapping trials from block that are also in remove_block. """
        delete_whole_n = []
        for n_block in range(0, len(self.blocks[block]['trial_windows'])):
            for n_rem_block in range(0, len(self.blocks[remove_block]['trial_windows'])):
                if self.blocks[remove_block]['trial_windows'][n_rem_block][1] < self.blocks[block]['trial_windows'][n_block][1]:
                    if self.blocks[block]['trial_windows'][n_block][0] < self.blocks[remove_block]['trial_windows'][n_rem_block][1]:
                        self.blocks[block]['trial_windows'][n_block][0] = self.blocks[remove_block]['trial_windows'][n_rem_block][1]
                if self.blocks[remove_block]['trial_windows'][n_rem_block][1] > self.blocks[block]['trial_windows'][n_block][1]:
                    if self.blocks[block]['trial_windows'][n_block][1] > self.blocks[remove_block]['trial_windows'][n_rem_block][0]:
                        self.blocks[block]['trial_windows'][n_block][1] = self.blocks[remove_block]['trial_windows'][n_rem_block][0]
                if self.blocks[block]['trial_windows'][n_block][0] == self.blocks[remove_block]['trial_windows'][n_rem_block][0]:
                    if self.blocks[block]['trial_windows'][n_block][1] == self.blocks[remove_block]['trial_windows'][n_rem_block][1]:
                        # This block is duplicate of remove block so delete it
                        delete_whole_n.append(n_block)
                        break

        for d_n in reversed(delete_whole_n):
            del self.blocks[block]['trial_windows'][d_n]

    def find_saccade_windows(self):
        """
            """
        # Force time_cushion to be integer so saccade windows are integers that can be used as indices
        time_cushion = np.array(self.sacc_time_cushion).astype('int')
        for trial in range(0, len(self.trials)):
            # Compute normalized eye speed vector and corresponding acceleration
            eye_speed = norm(np.vstack((self.trials[trial]['eye_velocity'][0], self.trials[trial]['eye_velocity'][1])), ord=None, axis=0)
            eye_acceleration = np.zeros(eye_speed.shape)
            eye_acceleration[1:] = np.diff(eye_speed, n=1, axis=0)

            # Find saccades as all points exceeding input thresholds
            threshold_indices = np.where(np.logical_or((eye_speed > self.velocity_thresh), (np.absolute(eye_acceleration) > self.acceleration_thresh)))[0]
            if threshold_indices.size == 0:
                # No saccades this trial
                self.trials[trial]['saccade_windows'] = np.empty(0).astype('int')
                self.trials[trial]['saccade_index'] = np.zeros(self.trials[trial]['duration_ms'], 'bool')
                continue

            # Find the end of all saccades based on the time gap between each element of threshold indices.
            # Gaps between saccades must exceed the time_cushion padding on either end to count as separate saccades.
            switch_indices = np.zeros_like(threshold_indices, dtype=bool)
            switch_indices[0:-1] =  np.diff(threshold_indices) > time_cushion * 2

            # Use switch index to find where one saccade ends and the next begins and store the locations.  These don't include
            # the beginning of first saccade or end of last saccade, so add 1 element to saccade_windows
            switch_points = np.where(switch_indices)[0]
            saccade_windows = np.full((switch_points.shape[0] + 1, 2), np.nan, dtype='int')

            # Check switch points and use their corresponding indices to find actual time values in threshold_indices
            for saccade in range(0, saccade_windows.shape[0]):
                if saccade == 0:
                    # Start of first saccade not indicated in switch points so add time cushion to the first saccade
                    # time in threshold indices after checking whether it is within time limit of trial data
                    if threshold_indices[saccade] - time_cushion > 0:
                        saccade_windows[saccade, 0] = threshold_indices[saccade] - time_cushion
                    else:
                        saccade_windows[saccade, 0] = 0

                # Making this an "if" rather than "elif" means in case of only 1 saccade, this statement
                # and the "if" statement above both run
                if saccade == saccade_windows.shape[0] - 1:
                    # End of last saccade, which is not indicated in switch points so add time cushion to end of
                    # last saccade time in threshold indices after checking whether it is within time of trial data
                    if len(eye_speed) < threshold_indices[-1] + time_cushion:
                        saccade_windows[saccade, 1] = len(eye_speed)
                    else:
                        saccade_windows[saccade, 1] = threshold_indices[-1] + time_cushion
                else:
                    # Add cushion to end of current saccade.
                    saccade_windows[saccade, 1] = threshold_indices[switch_points[saccade]] + time_cushion

                # Add cushion to the start of next saccade if there is one
                if saccade_windows.shape[0] > saccade + 1:
                    saccade_windows[saccade + 1, 0] = threshold_indices[switch_points[saccade] + 1] - time_cushion
            self.trials[trial]['saccade_windows'] = saccade_windows

            # Set boolean index for marking saccade times and save
            saccade_index = np.zeros(self.trials[trial]['duration_ms'], 'bool')
            for saccade_win in self.trials[trial]['saccade_windows']:
                saccade_index[saccade_win[0]:saccade_win[1]] = True
            self.trials[trial]['saccade_index'] = saccade_index

    def subtract_eye_offsets(self, name_params_dict, no_pos=False, no_vel=False, epsilon_eye=0.1, max_iter=10):
        """
            Only trials that appear in the name_params_dict will be adjusted. Each entry of name_params_dict
            is a trial name that returns a 3 element list whose ordered entries are: [0] - maestro event number to
            align on; [1] - maestro target to align on (e.g. fixation target = 0); [2] - a two element time window
            relative to the align event in which to take the mode eye data from subtraction (e.g. [-200, 0]). """


        print("WARNING: SOMETHING ABOUT THIS FUNCTION ISN'T RIGHT DUE TO THE RANDOMNESS OF WARNINGS FOR MEAN OF EMPTY SLICE (observed on data from Yoda_49)")
        delta_eye = np.zeros(len(self.trials))
        n_iters = 0
        next_trial_list = [x for x in range(0, len(self.trials))]
        while n_iters < max_iter and len(next_trial_list) > 0:
            trial_indices = next_trial_list
            next_trial_list = []
            for trial in trial_indices:
                curr_trial_name = self.trials[trial]['trial_name']
                if curr_trial_name not in name_params_dict:
                    continue
                if not self.trials[trial]['maestro_events'][name_params_dict[curr_trial_name][0]]:
                    # Current trial lacks desired align_segment
                    continue

                align_time = self.trials[trial]['targets'][name_params_dict[curr_trial_name][1]].get_next_refresh(self.trials[trial]['maestro_events'][name_params_dict[curr_trial_name][0]][0])
                time_index = np.arange(align_time + name_params_dict[curr_trial_name][2][0], align_time + name_params_dict[curr_trial_name][2][1] + 1, 1, dtype='int')
                if 'saccade_windows' in self.trials[trial]:
                    saccade_index = self.trials[trial]['saccade_index'][time_index]
                    time_index = time_index[~saccade_index]
                    if np.all(saccade_index):
                        # Entire fixation window has been marked as a saccade
                        # TODO This is an error, or needs to be fixed, or skipped??
                        print("Entire fixation window marked as saccade for trial {} and was skipped".format(trial))
                        continue
                if (not np.any(np.abs(self.trials[trial]['eye_velocity'][0][time_index]) < self.velocity_thresh) or
                    not np.any(np.abs(self.trials[trial]['eye_velocity'][1][time_index]) < self.velocity_thresh)):
                    # Entire fixation window is over velocity threshold
                    # TODO This is an error, or needs to be fixed, or skipped??
                    print("Entire fixation window marked as saccade for trial {} and was skipped".format(trial))
                    continue

                if not no_vel:
                    # Get modes of eye data in fixation window, excluding saccades and velocity times over threshold, then subtract them from eye data
                    horizontal_vel_mode, n_horizontal_vel_mode = mode(self.trials[trial]['eye_velocity'][0][time_index][np.abs(self.trials[trial]['eye_velocity'][0][time_index]) < self.velocity_thresh])
                    self.trials[trial]['eye_velocity'][0] = self.trials[trial]['eye_velocity'][0] - horizontal_vel_mode
                    vertical_vel_mode, n_vertical_vel_mode = mode(self.trials[trial]['eye_velocity'][1][time_index][np.abs(self.trials[trial]['eye_velocity'][1][time_index]) < self.velocity_thresh])
                    self.trials[trial]['eye_velocity'][1] = self.trials[trial]['eye_velocity'][1] - vertical_vel_mode
                else:
                    horizontal_vel_mode = 0
                    vertical_vel_mode = 0

                # Position mode is also taken when VELOCITY is less than threshold and adjusted by the difference between eye data and target position
                # if trial == 112:
                #     print(np.nanmean(self.trials[trial]['targets'][name_params_dict[curr_trial_name][1]].position[0, time_index]))
                #     print(self.trials[trial]['targets'][name_params_dict[curr_trial_name][1]].position[0, time_index])
                #     print(self.trials[trial]['eye_position'][0][time_index][np.abs(self.trials[trial]['eye_velocity'][0][time_index]) < self.velocity_thresh])
                #     # return
                # print(trial, self.trials[trial]['eye_velocity'][0][time_index].shape)

                if not no_pos:
                    sub_mean = np.nanmean(self.trials[trial]['targets'][name_params_dict[curr_trial_name][1]].position[0, time_index])
                    h_position_offset = self.trials[trial]['eye_position'][0][time_index][np.abs(self.trials[trial]['eye_velocity'][0][time_index]) < self.velocity_thresh] - sub_mean
                    horizontal_pos_mode, n_horizontal_pos_mode = mode(h_position_offset)
                    self.trials[trial]['eye_position'][0] = self.trials[trial]['eye_position'][0] - horizontal_pos_mode
                    v_position_offset = self.trials[trial]['eye_position'][1][time_index][np.abs(self.trials[trial]['eye_velocity'][1][time_index]) < self.velocity_thresh] - np.nanmean(self.trials[trial]['targets'][name_params_dict[curr_trial_name][1]].position[1, time_index])
                    vertical_pos_mode, n_vertical_pos_mode = mode(v_position_offset)
                    self.trials[trial]['eye_position'][1] = self.trials[trial]['eye_position'][1] - vertical_pos_mode
                else:
                    horizontal_pos_mode = 0
                    vertical_pos_mode = 0

                delta_eye[trial] = np.amax(np.abs((horizontal_vel_mode, vertical_vel_mode, horizontal_pos_mode, vertical_pos_mode)))
                if delta_eye[trial] >= epsilon_eye:
                    # This trial's offsets aren't good enough so try again next loop
                    next_trial_list.append(trial)

            self.find_saccade_windows()
            n_iters += 1

    def assign_retinal_motion(self, target_num):
        """ ."""
        for trial in self.trials:
            calculated_velocity = trial['targets'][target_num].velocity_from_position(True)
            trial['retinal_motion'] = calculated_velocity - trial['eye_velocity']

    def assign_acceleration(self, filter_win=10):
        """ Computes the acceleration from the velocity. """
        for trial in self.trials:
            trial['eye_acceleration'] = savgol_filter(trial['eye_velocity'], filter_win, 1, deriv=1, axis=1) * 1000


    def align_timeseries_to_event(self, name_event_num_dict, event_num=0, target_num=1, occurrence_n=0, next_refresh=True):
        """ Timeseries are defined starting at time 0 for current aligned event, this aligment subtracts the
            Plexon event time relative to trial time from each time series.
            The alignment event number is specified by the input dictionary name_event_num_dict
            that contains trial names and returns the alignment event number. Trial names not
            in this dictionary will NOT be aligned. If name_event_num_dict is an empty
            dictionary, all trials will be aligned on event event_num (default 0)"""

        if len(name_event_num_dict) == 0:
            for t in self.trials:
                if t['trial_name'] in name_event_num_dict:
                    continue
                name_event_num_dict[t['trial_name']] = event_num

        for t in range(0, len(self.trials)):
            if self.trials[t]['trial_name'] not in name_event_num_dict:
                continue
            event_number = name_event_num_dict[self.trials[t]['trial_name']]
            if event_number > len(self.trials[t]['maestro_events']):
                continue
            if occurrence_n > len(self.trials[t]['maestro_events'][event_number]) - 1:
                continue

            if next_refresh:
                # Find the difference between next refresh time and Maestro event time add it to Plexon event time
                align_time = (self.trials[t]['plexon_events'][event_number][occurrence_n] +
                              (self.trials[t]['targets'][target_num].get_next_refresh(self.trials[t]['maestro_events'][event_number][occurrence_n]) -
                               self.trials[t]['maestro_events'][event_number][occurrence_n]))
            else:
                align_time = self.trials[t]['plexon_events'][event_number][occurrence_n]

            # Adjust align time to be relative to trial start (using plexon STOP CODE!) rather than Plexon file time
            align_time = self.trials[t]['duration_ms'] - (self.trials[t]['plexon_start_stop'][1] - align_time)

            # Un-do old time series alignment, do new alignment and save it's time
            self.trials[t]['time_series'] += self.trials[t]['time_series_alignment']
            self.trials[t]['time_series'] -= align_time
            self.trials[t]['time_series_alignment'] = align_time
        # Update any neurons to new alignment
        for n in range(0, len(self.neurons)):
            self.neurons[n].align_spikes_to_session()

    def align_timeseries_to_latency(self):
        """ ."""
        pass

    def get_trial_index(self, trial_names, block_name=None, n_block_occurence=0):
        """. """
        if type(trial_names) == str:
            trial_names = [trial_names]
        if block_name is not None:
            search_range = self.blocks[block_name]['trial_windows'][n_block_occurence]
        else:
            search_range = (0, len(self.trials))
        trial_index = np.zeros(len(self.trials), dtype='bool')
        if trial_names is None:
            trial_index[search_range[0]:search_range[1]] = True
        else:
            for t in range(search_range[0], search_range[1]):
                for t_name in trial_names:
                    if self.trials[t]['trial_name'] == t_name:
                        trial_index[t] = True
        return trial_index

    def condition_data_by_trial(self, trial_name, time_window, data_type,
                                block_name=None, n_block_occurence=0):
        """. """
        trial_index = self.get_trial_index(trial_name, block_name, n_block_occurence)
        trial_data = np.full((np.count_nonzero(trial_index), time_window[1]-time_window[0], 2), np.nan)
        out_row = 0
        for t in range(0, len(self.trials)):
            if not trial_index[t]:
                continue
            time_index = self.trials[t]['time_series'].find_index_range(time_window[0], time_window[1])
            if time_index is None:
                # Entire trial window is beyond available data
                trial_index[t] = False
                continue
            if round(self.trials[t]['time_series'].start) - time_window[0] > 0:
                # Window start is beyond available data
                out_start = round(self.trials[t]['time_series'].start) - time_window[0]
            else:
                out_start = 0
            if round(self.trials[t]['time_series'].stop) - time_window[1] < 0:
                # Window stop is beyond available data
                out_stop = trial_data.shape[1] - (time_window[1] - round(self.trials[t]['time_series'].stop))
            else:
                out_stop = trial_data.shape[1]
            trial_data[out_row, out_start:out_stop, 0:2] = self.trials[t][data_type][:, time_index].T

            if self.nan_saccades:
                for start, stop in self.trials[t]['saccade_windows']:
                    # Align start/stop with desired offset
                    # start += nan_sacc_lag
                    # stop += nan_sacc_lag
                    # Align start/stop with time index and window
                    if stop > time_index[0] and stop < time_index[-1]:
                        start = max(start, time_index[0])
                    elif start > time_index[0] and start < time_index[-1]:
                        stop = min(stop, time_index[-1])
                    else:
                        continue
                    # Align start/stop with trial_data
                    start += out_start - time_index[0]
                    stop += out_start - time_index[0]
                    trial_data[out_row, start:stop, :] = np.nan
            out_row += 1
        trial_data = trial_data[0:out_row, :, :]
        return trial_data, trial_index

    def target_condition_data_by_trial(self, trial_name, time_window, data_type,
            target_number, block_name=None, n_block_occurence=0):
        """. """
        trial_index = self.get_trial_index(trial_name, block_name, n_block_occurence)
        target_data = np.empty((np.count_nonzero(trial_index), time_window[1]-time_window[0], 2))
        out_row = 0
        for t in range(0, len(self.trials)):
            if not trial_index[t]:
                continue
            time_index = self.trials[t]['time_series'].find_index_range(time_window[0], time_window[1])
            if time_index is None:
                # Entire trial window is beyond available data
                trial_index[t] = False
                continue
            if round(self.trials[t]['time_series'].start) - time_window[0] > 0:
                # Window start is beyond available data
                out_start = round(self.trials[t]['time_series'].start) - time_window[0]
            else:
                out_start = 0
            if round(self.trials[t]['time_series'].stop) - time_window[1] < 0:
                # Window stop is beyond available data
                out_stop = target_data.shape[1] - (time_window[1] - round(self.trials[t]['time_series'].stop))
            else:
                out_stop = target_data.shape[1]
            target_data[out_row, out_start:out_stop, 0:2] = getattr(self.trials[t]['targets'][target_number],
                                                                    data_type)[:, time_index].T
            out_row += 1
        target_data = target_data[0:out_row, :, :]
        return target_data, trial_index

    def nan_sacc_one_trial(trial_dict):
        pass
