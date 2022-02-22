import numpy as np



def find_trial_blocks(trial_data, trial_names, ignore_trial_names=[''], block_min=0, block_max=np.inf,
                      max_absent=0, max_consec_absent=np.inf, max_absent_pct=1, max_consec_single=np.inf):
    """ Finds the indices of blocks of trials satisfying the input criteria.
    The indices are given so that the number for BLOCK STOP IS ONE GREATER THAN
    THE ACTUAL TRIAL! so that it can be SLICED appropriately.

    Parameters
    ----------
    trial_data : python list of Trial type objects
        This can be the usual trial_data list. For the purposes of this function,
        it is only required to be a list of dictionaries, each containing the
        key 'name' which returns the name of the trial.
        name : string
            Name of the given trial that will be sought.
    trial_names : list of strings
        Each element of the list is a string of a trial name that should be
        included in the block.
    ignore_trial_names : list of strings
        Each element of the list is a string of a trial name that should be
        ignored even if they occur within a block of the desired trial names
        given in trial_names.
    block_min : int
        Minimum number of trials required to form a block.
    block_max : int
        Maximum number of trials allowed to form 1 block.
    max_absent : int
        Maximum number of trials within a block that are absent from the list
        of 'trial_names' and can still be considered the correct block.
    max_consec_absent : int
        Maximum number of trials within a block that are absent from the list
        of 'trial_names' consecutively and can still be considered the correct
        block.
    max_absent_pct : int
        Maximum number of trials within a block that are absent from the list
        of 'trial_names' and can still be considered the correct block.
    max_consec_single : int
        Maximum number of consecutive trials with an identical trial name that
        are allowed to be considered the same block

    Returns
    -------
    block_trial_windows : list of Window
        A list of Window objects indicating the beginning and end of each block
        found that satisfies the input criteria.
    """
    if (max_consec_absent < np.inf) and (max_consec_absent > 0):
        if max_absent < max_consec_absent:
            max_absent = np.inf

    block_trial_windows = []
    stop_triggers = []
    check_block = False
    final_check = False
    foo_block_start = None
    for trial in range(0, len(trial_data)):
        if trial_data[trial]['name'] in trial_names and not check_block:
            # Block starts
            # print("STARTED block {}".format(trial))
            n_absent = 0
            n_consec_absent = 0
            n_consec_single = 0
            check_block = True
            foo_block_start = trial
        elif trial_data[trial]['name'] in trial_names and check_block:
            # Good trial in the current block being checked
            # print("CONTINUE block {}".format(trial))
            n_consec_absent = 0
            if n_consec_single > max_consec_single:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_consec_single exceeded')
            else:
                if trial_data[trial]['name'] == last_trial_name:
                    n_consec_single += 1
                else:
                    n_consec_single = 0
            foo_block_stop = trial
        elif trial_data[trial]['name'] not in trial_names and trial_data[trial]['name'] not in ignore_trial_names and check_block:
            # Bad trial for block
            # print("FAILED block {}".format(trial))
            n_absent += 1
            if n_absent > max_absent:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_absent exceeded')
            elif n_consec_absent > max_consec_absent:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_consec_absent exceeded')
            # else:
            #     stop_triggers.append('good block')
            n_consec_absent += 1
        else:
            pass
        last_trial_name = trial_data[trial]['name']

        if trial == len(trial_data)-1:
            final_check = True
            this_trigger = (trial, 'end of file reached')

        if final_check:
            # Check block qualities that can't be done without knowing entire block
            # and then save block for output or discard
            # print("FINAL CHECK {}".format(trial))
            if foo_block_start is None:
                block_len = 0
            else:
                block_len = foo_block_stop - foo_block_start
            if block_len > 0 and check_block:
                # Subtract 1 from n_absent here because 1 was added to it just to break the block and is not included in it
                if block_len >= block_min and block_len <= block_max and ((n_absent - 1)/ block_len) <= max_absent_pct:
                    block_trial_windows.append(Window([foo_block_start, foo_block_stop + 1]))
                    stop_triggers.append(this_trigger)
            check_block = False
            final_check = False

    return block_trial_windows, stop_triggers


class ConjoinedMismatchError(Exception):
    # Dummy exception class for a potential problem with list matching
    pass


class Window(object):
    """ Very simple class indicating that two numbers should be treated as a
    window, or range, for creating indices with step size 1. """
    def __init__(self, start=None, stop=None):
        if start is None:
            raise ValueError("No values input for start.")
        try:
            # Check if start has more than 1 element via len()
            len(start)
            is_num = False # Not a num if it has len
        except TypeError:
            is_num = True
        if not is_num:
            if (len(start) == 2) and (stop is None):
                self.start = int(start[0])
                self.stop = int(start[1])
            elif (len(start) >= 2) and (stop is not None):
                raise ValueError("Too many values input for start and stop.")
            elif (len(start) == 1) and (stop is None):
                raise ValueError("Must input values for both start and stop.")
            else:
                raise ValueError("Unrecognized start stop contingency")
        else:
            self.start = int(start)
            self.stop = int(stop)
        if self.stop <= self.start:
            raise ValueError("Input stop window must be greater than start. Current start = {0} and stop = {1}.".format(self.start, self.stop))

    def __getitem__(self, index):
        if index == 0:
            return self.start
        elif index == 1:
            return self.stop
        else:
            raise IndexError("Window objects have 2 indices, 0 and 1, for start and stop but index '{0}' was requested.".format(index))

    def __len__(self):
        return 2

    def __repr__(self):
        return "[{0}, {1}]".format(self.start, self.stop)


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

    def __iter__(self):
        return self.__list__.__iter__()

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


class Session(object):
    """ A class containing trials and their associated timeseries and events
    to allow behavioral and neural analysis of specific trials from an
    experimental session.

    The base object should be created from a list of dictionaries that represent
    trials. From this point, list of neuron dictionaries can be added as can
    easier access labels such as trial blocks. The data for trial name, events,
    and timeseries from the first input trial_data will be used as metadata and
    enforced across all trial lists that attempt to join the session.

    The list of trials is added as a ConjoinedList so that lists of neurons can
    be added to the behavioral trial data. Each trial can contain events and
    multiple named timeseries data so that behavior and neuron analysis can be
    performed. Trials can contain names for later reference and indexing as well
    as blocks, which index groups of related trials.

    Parameters
    ----------
    trial_data : python list of Trial type objects
        Each element of the list must contain an object of the Trial class
        specified in the trial.py module. This is not explicitly checked
        because double imports can confuse checking so undefined errors could
        result if other types are used. The first list of trial objects will
        be treated as the parent for any future trial sets added. This list
        will also be used to generate metadata, like trial name, and
        timeseries that will be enforced across new trials added to session.
        name : string
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
        aligned_event : (optional) var
            A key into the 'events' dictionary indicating the event on which
            the trial is currently aligned.
    session_name : string
        String naming the session for user reference. Default = None.

    Returns
    -------
    None :
        Creates an instantiation of the Session class.
    """

    def __init__(self, trial_data, session_name=None, data_type=None):
        """
        """
        self._trial_lists = {}
        self._trial_lists['__main'] = ConjoinedList(trial_data)
        if data_type is None:
            # Use data name of first trial as default
            data_type = trial_data[0].__data_alias__
        # The first data listed is hard coded here as the parent of conjoined lists
        self._trial_lists[data_type] = self._trial_lists['__main']
        self.__series_names = {}
        self._session_trial_data = []
        for t in trial_data:
            st = {}
            for k in t['data'].keys():
                self.__series_names[k] = data_type

            # Extract some metadata that will be enforced over all trial sets
            # joining this session
            st['name'] = t['name']
            st['events'] = t['events']
            try:
                st['aligned_event'] = t['aligned_event']
            except KeyError:
                st['aligned_event'] = None
            st['aligned_time'] = t._timeseries[0]
            st['timeseries'] = t._timeseries
            st['curr_t_win'] = {}
            st['curr_t_win']['time'] = [-np.inf, np.inf]
            self._session_trial_data.append(st)

        self._session_trial_data = ConjoinedList(self._session_trial_data)
        self._trial_lists['__main'].add_child(self._session_trial_data)
        self.session_name = session_name
        self.blocks = {}
        self.neurons = []

    def add_trial_data(self, trial_data, data_type=None):
        """ Adds a new list of trial dictionaries that will be conjoined with
        the existing list initialized via __init__. """
        if data_type is None:
            # Use data name of first trial as default
            data_type = trial_data[0].__data_alias__
        if data_type in self._trial_lists:
            raise ValueError("Session already has data type {0}.".format(data_type))
        self.__validate_trial_data(trial_data)
        # Check to update data_names so we can find their associated list of
        # trials quickly later (e.g. in get_data)
        new_names = set()
        for t in trial_data:
            for k in t['data'].keys():
                new_names.add(k)
        for nn in new_names:
            if nn in self.__series_names.keys():
                raise ValueError("Session already has data name {0}.".format(nn))

        for nn in new_names:
            self.__series_names[nn] = data_type
        # Save dictionary reference to this trial set and conjoin to __main
        self._trial_lists[data_type] = ConjoinedList(trial_data)
        self._trial_lists['__main'].add_child(self._trial_lists[data_type])

        return None

    def get_data_list(self, series_name, trials, time):
        """ Returns a list of length trials, where each element of the list
        contains the timeseries data in the requested time window. If the time
        window exceeds the valid range of the timeseries, data are excluded.
        Thus the output list is NOT necessarily in 1-1 correspondence with the
        input trials sequence!
        Call "data_names()" to get a list of available data names. """
        data_out = []
        data_name = self.__series_names[series_name]
        t_inds = self.__parse_trials_to_indices(trials)
        for t in t_inds:
            trial_obj = self._trial_lists[data_name][t]
            trial_ts = trial_obj._timeseries
            try:
                trial_tinds = trial_ts.find_index_range(time[0], time[1])
            except IndexError:
                continue
            data_out.append(trial_obj['data'][series_name][trial_tinds])

        return data_out

    def get_data_array(self, series_name, trials, time):
        """ Returns a n trials by m time points numpy array of the requested
        timeseries data. Missing data points are filled in with np.nan.
        Call "data_names()" to get a list of available data names. """
        data_out = []
        data_name = self.__series_names[series_name]
        t_inds = self.__parse_trials_to_indices(trials)
        for t in t_inds:
            trial_obj = self._trial_lists[data_name][t]
            self._set_t_win(t, time)
            valid_tinds = self._session_trial_data[t]['curr_t_win']['valid_tinds']
            out_inds = self._session_trial_data[t]['curr_t_win']['out_inds']
            t_data = np.full(out_inds.shape[0], np.nan)
            t_data[out_inds] = trial_obj['data'][series_name][valid_tinds]
            data_out.append(t_data)

        return np.vstack(data_out)

    def _get_blocks(self, block_name):
        """Tries to get the trial indices associated with the specified block
        name string and throws a more appropriate error during failure. """
        try:
            t_block = self.blocks[block_name]
        except KeyError:
            raise KeyError("Session object has no block named {0}.".format(block_name))
        return t_block

    def __parse_trials_to_indices(self, trials):
        """ Can accept string inputs indicating block names, or slices of indices,
        or numpy array of indices, or list of indices and outputs a corresponding
        useful index for getting the corresponding trials."""
        if type(trials) == slice:
            # Return the slice of all trial indices
            t_inds = np.arange(0, len(self), dtype=np.int32)
            t_inds = t_inds[trials]
        elif (type(trials) == np.ndarray) or (type(trials) == list):
            t_inds = np.int32(trials)
        elif type(trials) == Window:
            t_inds = np.arange(trials[0], trials[1], dtype=np.int32)
        elif type(trials) == str:
            trials = self._get_blocks[trials]
            if type(trials) == str:
                raise RuntimeError("Input string for trials returned a block string which will result in multiple recursion. Check blocks attribute for errors.")
            t_inds = self.__parse_trials_to_indices(trials)
        return t_inds

    def __parse_time_to_indices(self, time):
        """ Accepts a 2 element time window, slice of times or array/list of
        times and turns them into the corresponding indices using the
        timeseries of the trial. """
        pass

    def data_names(self):
        """Provides a list of the available data names. """
        return [x for x in self._trial_lists.keys() if x[0:2] != "__"]

    def series_names(self):
        """Provides a list of the available data series names under the given
        data_name. """
        return [x for x in self.__series_names.keys()]

    def _set_t_win(self, trial_ind, t_win):
        # First check if this is even a new t_win before updating
        if ( (self._session_trial_data[trial_ind]['curr_t_win']['time'][0] == t_win[0])
            and (self._session_trial_data[trial_ind]['curr_t_win']['time'][1] == t_win[1]) ):
            # No change in t_win
            return None
        trial_ts = self._session_trial_data[trial_ind]['timeseries']
        valid_tinds, out_inds = trial_ts.valid_index_range(t_win[0], t_win[1])
        self._session_trial_data[trial_ind]['curr_t_win']['time'] = t_win
        self._session_trial_data[trial_ind]['curr_t_win']['valid_tinds'] = valid_tinds
        self._session_trial_data[trial_ind]['curr_t_win']['out_inds'] = out_inds
        return None

    def __validate_trial_data(self, trial_data):
        """ Checks data in trial_data against imeseries data of
        the main trial list and metadata to ensure newly added trials match
        the timeseries data and length. """

        for t_ind, t in enumerate(trial_data):
            if t['name'] != self._session_trial_data[t_ind]['name']:
                raise ValueError("New trial data in trial number {0} trial name does not matching existing session trial name ({1} vs. {2}).".format(t_ind, t['name'], self._session_trial_data[t_ind]['name']))
            if len(t._timeseries) != len(self._session_trial_data[t_ind]['timeseries']):
                raise ValueError("Length of new trial data in trial number {0} does not match existing session trial.".format(t_ind))
            if t._timeseries[0] != self._session_trial_data[t_ind]['timeseries'][0]:
                raise ValueError("New trial data in trial number {0} is not aligned to the same time for index zero as existing session trial ({1} vs. {2}).".format(t_ind, t._timeseries[0], self._session_trial_data[t_ind][0]))

        return None

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

    def add_blocks(self, trial_names, block_names, **criteria):
        """ Adds groups of trials to blocks named 'block_names'. Blocks are
        named in order of trial number using the order of names provided in
        block_names. If more blocks are found than block_names given, and error
        is raised. If fewer blocks are found than names given, the extra names
        are ignored.
        """
        trial_windows, _ = find_trial_blocks(self._trial_lists['__main'], trial_names, **criteria)
        if type(block_names) == list:
            if len(trial_windows) > len(block_names):
                raise RunTimeError("Found more blocks than names given. Add block_names or check that criteria are appropriately strict.")
            for tw_ind in range(0, len(trial_windows)):
                self.blocks[block_names[tw_ind]] = trial_windows[tw_ind]
        else:
            if len(trial_windows) > 1:
                raise RunTimeError("Found more blocks than names given. Add block_names or check that criteria are appropriately strict.")
            else:
                self.blocks[block_name] = trial_windows[0]
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

    def __len__(self):
        return len(self._trial_lists['__main'])

    def __getitem__(self, item):
        if type(item) == str:
            try:
                return self._trial_lists[item]
            except KeyError:
                raise ValueError("Could not find data name '{0}'. Use method 'data_names()' for list of valid data names".format(item))
        elif (type(item) == int) or (type(item) == slice):
            return self._trial_lists['__main'][item]
        else:
            raise ValueError("Cannot find item '{0}' in session".format(item))
