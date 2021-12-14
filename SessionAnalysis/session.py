import numpy as np
from scipy.stats import mode
from scipy.signal import savgol_filter
from numpy.linalg import norm



class ConjoinedMismatchError(Exception):
    # Dummy exception class for a potential problem with list matching
    pass

from collections.abc import MutableSequence
class ConjoinedList(MutableSequence):
    """ Objects of this class essentially behave as lists for purposes of storing,
        indexing and iterating. Their key feature is they can be conjoined to
        other ConjoinedList objects to enforce both lists to remain in a one-to-one
        index mapping.  If elements from one list are deleted, the same elements
        are deleted from all of its conjoined lists. There is a weak hierarchy
        insisting that new child ConjoinedLists are not conjoined to any other
        lists. The children are still able to enforce one-to-one mapping
        with their parent ConjoinedList.

        Note that if you attempt to delete a reference to an instance of a
        ConjoinedList, references to it will remain in any other ConjoinedLists
        to which it was previously joined.

        Parameters
        ----------
        data_list : python list

        Returns
        -------
        ConjoinedList :
            Creates an instantiation of the ConjoinedList class.
    """
    def __init__(self, data_list):
        self._initialized_list = False
        self.__list__ = list(data_list)
        # Conjoined list is conjoined with itself
        self._conjoined_lists = [self]
        self._element_ID = [x for x in range(0, len(data_list))]
        self._next_ID = len(data_list)
        self._initialized_list = True

    def __delitem__(self, index):
        print("DELETING INDEX", index)
        for cl in self._conjoined_lists:
            cl.__non_recursive_delitem__(index)
        self.__check_match_IDs__()
        return

    def __delete__(self):
        print("IN THE DELETE")

    def __del__(self):
        print("IN THE DEL")

    def __non_recursive_delitem__(self, index):
        del self.__list__[index]
        del self._element_ID[index]
        return None

    def insert(self, index, values):
        """ Values will be inserted first to the current list, then each list in
        _conjoined_lists in order. """
        if len(values) != len(self._conjoined_lists):
            raise ValueError("A value must be inserted for each conjoined list in _conjoined_lists.")
        else:
            for cl_ind, cl in enumerate(self._conjoined_lists):
                cl.__list__.insert(index, values[cl_ind])
                cl._element_ID.insert(index, cl._next_ID)
                cl._next_ID += 1
        self.__check_match_IDs__()
        return None

    def __setitem__(self, index, value):
        self.__list__[index] = value
        return None

    def __getitem__(self, index):
        return self.__list__[index]

    def __len__(self):
        return len(self.__list__)

    def sort(self, key=None, reverse=False):
        new_order = [y for x, y in sorted(zip(self.__list__,
                        [x for x in range(0, len(self.__list__))]), key=key)]
        if reverse:
            new_order = [x for x in reversed(new_order)]
        self.order(new_order)
        # Match length is checked in "self.order"
        return None

    def order(self, new_order):
        """ Reorders all conjoined lists according to the order indices input
        in new_order. """
        for cl in self._conjoined_lists:
            cl.__non_recursive_order__(new_order)
        self.__check_match_IDs__()
        return None

    def __non_recursive_order__(self, new_order):
        self.__list__ = [self.__list__[x] for x in new_order]
        self._element_ID = [self._element_ID[x] for x in new_order]
        return None

    def add_child(self, ChildConjoinedList):
        """ Attach another ConjoinedList object to the current one. This new
        child list cannot be associated with other ConjoinedList objects and
        will inherit the _element_ID properties of the current list. Enforcing
        this weak sense of hiearchy prevents chaotic conjoining and
        rearranging. """
        if len(ChildConjoinedList) != len(self):
            raise ValueError("Additional lists must be of the same length!")
        if len(ChildConjoinedList._conjoined_lists) > 1:
            raise AttributeError("Cannot conjoin with Conjoined lists that have been previously conjoined to other lists.")
        self._conjoined_lists.append(ChildConjoinedList)
        # This child conjoined list inherits the matching IDs of the parent
        ChildConjoinedList._element_ID = [x for x in self._element_ID]
        try:
            # Child conjoined list is conjoined to every list in parent
            for cl in self._conjoined_lists:
                if cl not in ChildConjoinedList._conjoined_lists:
                    ChildConjoinedList._conjoined_lists.append(cl)
        except AttributeError as err:
            raise AttributeError("Only ConjoinedList type can be conjoined.") from err
        return None

    def __str__(self):
        return f"{self.__list__}"

    def __repr__(self):
        return f"SessionAnalysis.ConjoinedList({self.__list__})"

    def __check_match_IDs__(self):
        for cl in self._conjoined_lists:
            if len(cl) != len(self):
                raise ConjoinedMismatchError("Conjoined lists no longer have same length!")
            if len(cl._element_ID) != len(set(cl._element_ID)):
                raise ConjoinedMismatchError("Conjoined list contains duplicate element IDs and may have been compromised!")
            for ind, ID in enumerate(self._element_ID):
                if ID != cl._element_ID[ind]:
                    raise ConjoinedMismatchError("Match between elements of conjoined lists has been broken!")
        return None


class Trial(dict):
    """ A class containing trials and their associated timeseries and events
    to allow behavioral and neural analysis of specific trials from an
    experimental session.

    The base object should be created from a list of dictionaries that represent
    trials. From this point, list of neuron dictionaries can be added as can
    easier access labels such as trial blocks.

    The list of trials is added as a ConjoinedList so that lists of neurons can
    be added to the behavioral trial data. Each trial can contain events and
    multiple named timeseries data so that behavior and neuron analysis can be
    performed. Trials can contain names for later reference and indexing as well
    as blocks, which index groups of related trials.

    Parameters
    ----------
    trial_dict : python dict
        Each dictionary must specifiy the following trial attributes via its
        keys to define a Trial. These keys are NOT CASE SENSITIVE. These keys
        will be mapped to lower case versions. Any additional keys will remain
        exactly as input. Integer keys are NOT allowed as the trial data can be
        indexed by integers and slices. If duplicate keys are used within the
        data dict and the events dict, the keys for the data dict are searched
        first. Any additional keys are not checked by the builtin/inherited
        functions (e.g. __getitem__, __delitem___ etc.)
        name : string
            Name of the given trial. Will be used to reference trials of this
            type.
        data : dict
            Dictionary containing time dependent data for the current trial,
            such as stimulus and/or behavioral movement data. Each key of the
            dictionary specifies a timeseries. The key names for this
            dictionary can be used later to reference and extract their
            corresponding data. The data referenced by each key should be
            indexable using integers or slices and will be converted to a
            numpy array.
        events : dict
            Dictionary where each key names an event. Each event is a time
            within the session on which the timeseries data (neural and
            behavioral data) can be aligned. Event times should therefor be in
            agreement with the time scheme of the timeseries data, i.e. starting
            the beginning of each trial t=0 or relative to the entire session.
            Event names can be used to reference events for alignment.

    Returns
    -------
    None :
        Creates an instantiation of the Trial class.
    """

    def __init__(self, trial_dict):
        """
        """
        # Keys required for trial dictionaries
        self.__required_trial_keys__ = ["name", "data", "events"]
        self._set_trial_data(trial_dict)
        self._check_trial_data()
        # For defining iterator over attributes
        self.__build_iterator__()

    def _set_trial_data(self, trial_dict):
        if type(trial_dict) != dict:
            raise InputError("Input trial_dict must be a dictionary.")

        rtk_copy = [x for x in self.__required_trial_keys__]
        for tdk in trial_dict.keys():
            # Check for required keys and assign them as lower case to self
            tdk_required = False
            for rk_ind, rk in enumerate(rtk_copy):
                if tdk.lower() == rk:
                    del rtk_copy[rk_ind]
                    if rk == "name":
                        self.name = trial_dict[tdk]
                        tdk_required = True
                        break
                    elif rk == "events":
                        self.events = trial_dict[tdk]
                        tdk_required = True
                        break
                    elif rk == "data":
                        self.data = trial_dict[tdk]
                        tdk_required = True
                        break
                    else:
                        # This shouldn't happen unless __required_trial_keys__ is out of date?
                        raise ValueError("Required key matched but not specified during parse")
            if not tdk_required:
                # This is a key that was input but not required
                setattr(self, tdk, trial_dict[tdk])

        # Check input now that we know the number of required keys found
        if len(rtk_copy) > 0:
            raise KeyError("Input trial_dict must have all required keys. Key(s) '{0}' not found.".format(rtk_copy))
        for rk in self.__required_trial_keys__:
            if rk == "name":
                if type(trial_dict[rk]) != str:
                    raise ValueError("Input trial_dict key 'name' must be a string of the trial name.")
            elif rk == "data":
                if type(trial_dict[rk]) != dict:
                    raise ValueError("Input trial_dict key 'data' must be a dictionary that maps data name/type to the data timeseries.")
            elif rk == "events":
                if type(trial_dict[rk]) != dict:
                    raise ValueError("Input trial_dict key 'events' must be a dictionary that maps event name to the time of the event.")
        return None

    def _check_trial_data(self):
        if len(self.data) > 0:
            for data_key in self.data.keys():
                print("Should set to timeseries or Numpy array in _set_trial_data")
        else:
            # No trial data present
            pass

    def __getitem__(self, index):
        if type(index) == tuple:
            # Multiple attribute/indices input so split
            attribute = index[0]
            index = index[1:]
            if len(index) == 1: index = index[0]
        else:
            attribute = index
            index = None
        if (type(index) == int) or (type(index) == slice) or (type(index) == tuple):
            is_index = True
        else:
            is_index = False
        if (index is None) and (type(attribute) == str):
            try:
                att_value = getattr(self, attribute)
                return att_value
            except AttributeError:
                for key in self.data.keys():
                    if key == attribute:
                        return self.data[key]
                for key in self.events.keys():
                    if key == attribute:
                        return self.events[key]
                raise ValueError("Could not find value '{0}'.".formate(attribute))
        elif (type(index) == str) and (type(attribute) == str):
            try:
                att_value = getattr(self, attribute)
                return att_value[index]
            except KeyError:
                raise ValueError("Could not find '{0}' in dictionary attribute '{1}'".format(index, attribute))
        elif (is_index) and (type(attribute) == str):
            try:
                att_value = getattr(self, attribute)
                return att_value[index]
            except AttributeError:
                if attribute in self.data:
                    return self.data[attribute][index]
                raise
        else:
            print("INDEXING CONTINGENCY NOT FOUND")
            pass

    def __contains__(self, value):
        if (type(value) == int) or (type(value) == slice):
            return False # Integer keys used as indices and not allowed
        try:
            self.__getitem__(value)
            return True
        except:
            return False

    def __delattr__(self, attribute):
        if attribute in self.__required_trial_keys__:
            raise ValueError("Cannot delete attribute '{0}' from objects of Trial class.".format(attribute))
        else:
            super().__delattr__(attribute)
        return None

    def __delitem__(self, item):
        if item in self.__required_trial_keys__:
            raise ValueError("Cannot delete item '{0}' from objects of Trial class.".format(item))
        try:
            self.__delattr__(item)
            return None
        except AttributeError:
            if item in self.data.keys():
                del self.data[item]
                return None
            elif item in self.events.keys():
                del self.events[item]
                return None
            else:
                raise AttributeError("No field '{0}' found in Trial data or events.".format(item))

    def __build_iterator__(self):
        iter_str = [x for x in self.__dict__.keys()]
        for x in ["data", "events"]:
            for key in self[x]:
                iter_str.append(key)
        self.__myiterator__ = iter(iter_str)

    def __iter__(self):
        # Reset iter object to current attributes
        self.__build_iterator__()
        return self

    def __next__(self):
        next_item = next(self.__myiterator__)
        # Do not return 'hidden' attributes/keys
        while next_item[0:2] == "__":
            next_item = next(self.__myiterator__)
        return next_item

    def __len__(self):
        n = 0
        for x in self:
            n += 1
        return n

    # def __setattr__(self):
    #     print("trying to set")

    def __clear__(self):
        pass

    def __update__(self):
        print("updating")

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Session(dict):
    """ A class containing trials and their associated timeseries and events
    to allow behavioral and neural analysis of specific trials from an
    experimental session.

    The base object should be created from a list of dictionaries that represent
    trials. From this point, list of neuron dictionaries can be added as can
    easier access labels such as trial blocks.

    The list of trials is added as a ConjoinedList so that lists of neurons can
    be added to the behavioral trial data. Each trial can contain events and
    multiple named timeseries data so that behavior and neuron analysis can be
    performed. Trials can contain names for later reference and indexing as well
    as blocks, which index groups of related trials.

    Parameters
    ----------
    trial_data : python list
        Each element of the list must contain a dictionary that specifies the
        following trial attributes via its keys. These keys and data are
        required to define a session.
        trial_name : string
            Name of the given trial. Will be used to reference trials of this
            type.
        behavioral_data : dict
            Dictionary containing behavioral data for the current trial. Each
            key of the dictionary specifies a behavioral timeseries. The key
            names for this dictionary can be used later to reference and extract
            their corresponding data.
        events : dict
            Dictionary where each key names an event. Each event is a time
            within the session on which the timeseries data (neural and
            behavioral data) can be aligned. Event times should therefor be in
            agreement with the time scheme of the timeseries data, i.e. starting
            the beginning of each trial t=0 or relative to the entire session.
            Event names can be used to reference events for alignment.
    session_name : string
        String naming the session for user reference. Default = None.

    Returns
    -------
    None :
        Creates an instantiation of the Session class.
    """

    def __init__(self, trial_data, session_name=None):
        """
        """
        self.set_trial_data(trial_data)
        self.session_name = session_name
        self.blocks = {}
        self.neurons = []
        # Keys required for trial dictionaries
        self._required_trial_keys = ["trial_name", "behavioral_data", "events"]

    def set_trial_data(self, trial_data):
        for t_ind, t in enumerate(trial_data):
            if type(t) != dict:
                raise ValueError("Each element of input trial data list must be a dictionary.")
            for rk in self._required_trial_keys:
                if rk not in t.keys():
                    raise ValueError("Dictionaries for each trial must have all required keys. Key ", rk, "not found in trial_data element ", t_ind)
        self.trial_data = ConjoinedList(trial_data)
        return None

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
