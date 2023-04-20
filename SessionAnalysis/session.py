import numpy as np



def find_trial_blocks(trial_data, trial_names, ignore_trial_names=[''], block_min=0, block_max=np.inf,
                      max_absent=0, max_consec_absent=np.inf, max_absent_pct=1, max_consec_single=np.inf,
                      n_min_per_trial=0):
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
    n_min_per_trial : int, list, or np.array of int
        The minimum number of each corresponding trial name in trial_names that
        must be found for a block to be considered a valid block.

    Returns
    -------
    block_trial_windows : list of Window
        A list of Window objects indicating the beginning and end of each block
        found that satisfies the input criteria.
    """
    if not isinstance(trial_names, list):
        raise ValueError("trial_names must be a list of trial names.")
    if not isinstance(ignore_trial_names, list):
        raise ValueError("ignore_trial_names must be a list of trial names.")
    if (max_consec_absent < np.inf) and (max_consec_absent > 0):
        if max_absent < max_consec_absent:
            max_absent = np.inf
    if not isinstance(n_min_per_trial, list):
        n_min_per_trial = [n_min_per_trial]
    if len(n_min_per_trial) != len(trial_names):
        if len(n_min_per_trial) == 1:
            n_min_per_trial = [n_min_per_trial[0] for x in range(0, len(trial_names))]
    t_name_inds = {}
    n_t_name = {}
    for tn_ind, t_name in enumerate(trial_names):
        n_t_name[t_name] = 0
        t_name_inds[t_name] = tn_ind

    block_trial_windows = []
    stop_triggers = []
    check_block = False
    final_check = False
    foo_block_start = None
    for trial in range(0, len(trial_data)):
        if trial_data[trial]['name'] in trial_names and not check_block:
            # Block starts
            n_absent = 0
            n_consec_absent = 0
            n_consec_single = 0
            check_block = True
            foo_block_start = trial
            foo_block_stop = trial
            # Reset trial name counts to zero
            for t_name in trial_names:
                n_t_name[t_name] = 0
            # Count current trial
            n_t_name[trial_data[trial]['name']] += 1
        elif trial_data[trial]['name'] in trial_names and check_block:
            # Good trial in the current block being checked
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
            n_t_name[trial_data[trial]['name']] += 1
        elif trial_data[trial]['name'] not in trial_names and trial_data[trial]['name'] not in ignore_trial_names and check_block:
            # Bad trial for block
            n_absent += 1
            if n_absent > max_absent:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_absent exceeded')
            elif n_consec_absent > max_consec_absent:
                # Block ends
                final_check = True
                this_trigger = (trial, 'n_consec_absent exceeded')
            else:
                pass
                # stop_triggers.append('good block')
            n_consec_absent += 1
        else:
            pass
        last_trial_name = trial_data[trial]['name']

        if trial == len(trial_data)-1:
            final_check = True
            this_trigger = (trial, 'end of file reached')
            foo_block_stop = trial

        if final_check:
            # Check block qualities that can't be done without knowing entire block
            # and then save block for output or discard
            # print("FINAL CHECK {}".format(trial))
            if foo_block_start is None:
                block_len = 0
            else:
                block_len = foo_block_stop - foo_block_start
            if block_len > 0 and check_block:
                # Check for enough of every trial name
                n_min_reached = False
                for t_name in trial_names:
                    if n_t_name[t_name] >= n_min_per_trial[t_name_inds[t_name]]:
                        n_min_reached = True
                    else:
                        n_min_reached = False
                        break

                # Subtract 1 from n_absent here because 1 was added to it just to break the block and is not included in it
                if ( (block_len >= block_min) and
                     (block_len <= block_max) and
                     (((n_absent - 1)/block_len) <= max_absent_pct) and
                     (n_min_reached) ):
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
        self.__check_values__()

    def __check_values__(self):
        if self.stop <= self.start:
            raise ValueError("Input stop window must be greater than start. Current start = {0} and stop = {1}.".format(self.start, self.stop))

    def __getitem__(self, index):
        if index == 0:
            return self.start
        elif index == 1:
            return self.stop
        else:
            raise IndexError("Window objects have 2 indices, 0 and 1, for start and stop but index '{0}' was requested.".format(index))

    def __setitem__(self, index, value):
        if index == 0:
            self.start = value
        elif index == 1:
            self.stop = value
        else:
            raise IndexError("Window objects have 2 indices, 0 and 1, for start and stop but index '{0}' was assigned.".format(index))
        self.__check_values__()

    def __iter__(self):
        return iter([self.start, self.stop])

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

    def __setitem__(self, index, value):
        self.__list__[index] = value
        return None

    def __getitem__(self, index):
        return self.__list__[index]

    def __len__(self):
        return len(self.__list__)

    def __iter__(self):
        return self.__list__.__iter__()

    def __non_recursive_order__(self, new_order):
        self.__list__ = [self.__list__[x] for x in new_order]
        self._element_ID = [self._element_ID[x] for x in new_order]
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
    as blocks, which index groups of related trials. Added groups of trials
    will be assigned the main timeseries from the first parent group of trials.
    The ideal parent group is a set of MaestroApparatusTrial objects.

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
            if st['aligned_time'] < len(t._timeseries):
                st['incl_align'] = True
            else:
                st['incl_align'] = False
            st['timeseries'] = t._timeseries
            st['curr_t_win'] = {}
            st['curr_t_win']['time'] = [-np.inf, np.inf]
            self._session_trial_data.append(st)

        self._session_trial_data = ConjoinedList(self._session_trial_data)
        self._trial_lists['__main'].add_child(self._session_trial_data)
        self.session_name = session_name
        self.blocks = {}
        self.trial_sets = {}
        self.__del_trial_set__ = True

    def add_trial_data(self, trial_data, data_type=None):
        """ Adds a new list of trial dictionaries that will be conjoined with
        the existing list initialized via __init__. New trials are assigned the
        same timeseries as the __main trials."""
        if data_type is None:
            # Use data name of first trial as default
            data_type = trial_data[0].__data_alias__
        if data_type in self._trial_lists:
            raise ValueError("Session already has data type {0}.".format(data_type))
        self.__validate_trial_data(trial_data)
        # Check to update data_names so we can find their associated list of
        # trials quickly later (e.g. in get_data_array)
        new_names = set()
        for t_ind, t in enumerate(trial_data):
            # Re-assign timeseries data
            t._timeseries = self._session_trial_data[t_ind]['timeseries']
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

    def add_data_series(self, trial_data_type, new_series_name):
        """ Updates the session object in the event that you add a data series
        to trial objects after initiating the session, such as retinal slip.
        The data series name will be registered with the session so it can find
        it.
        """
        if trial_data_type not in self._trial_lists:
            raise ValueError("Session does not contrain trial objects of input type {0} so new series cannot be added to it.".format(trial_data_type))
        if new_series_name in self.__series_names.keys():
            raise ValueError("Session already has data series name {0}.".format(new_series_name))
        self.__series_names[new_series_name] = trial_data_type


    def add_neuron_trials(self, trial_data, neuron_meta, meta_dict_name='meta_data'):
        """ Adds a new list of trial dictionaries that will be conjoined with
        the existing list initialized via __init__. New trials are assigned the
        same timeseries as the __main trials. Reserved specifically for neurons
        with data_type "neurons" for tracking class ID of neurons and maybe
        other things like being absent on particular trials searching a neuron
        metadata. Input a dictionary of Neuron objects in 'neuron_meta' that
        will be used for deciding good trials, tuning, etc. """
        data_type = "neurons"
        if data_type in self._trial_lists:
            raise ValueError("Session already has data type {0}.".format(data_type))
        # Store reference to neuron meta data dictionaries for each trial
        self.meta_dict_name = meta_dict_name
        # Set global dictionaries for each neuron found for properties that are
        # session-wide for neurons
        self.neuron_info = neuron_meta
        self.__validate_trial_data(trial_data)
        # Check to update data_names so we can find their associated list of
        # trials quickly later (e.g. in get_data_array)
        new_names = set()
        for t_ind, t in enumerate(trial_data):
            # Re-assign timeseries data
            if t._timeseries.start != self._session_trial_data[t_ind]['timeseries'].start:
                raise ValueError("Input data timeseries start time {0} does not match existing session timeseries start time {1}.".format(t._timeseries.start, self._session_trial_data[t_ind]['timeseries'].start))
            if t._timeseries.stop != self._session_trial_data[t_ind]['timeseries'].stop:
                raise ValueError("Input data timeseries start time {0} does not match existing session timeseries start time {1}.".format(t._timeseries.stop, self._session_trial_data[t_ind]['timeseries'].stop))
            if t._timeseries.dt != self._session_trial_data[t_ind]['timeseries'].dt:
                raise ValueError("Input data timeseries start time {0} does not match existing session timeseries start time {1}.".format(t._timeseries.dt, self._session_trial_data[t_ind]['timeseries'].dt))
            t._timeseries = self._session_trial_data[t_ind]['timeseries']
            if "dt" not in self.neuron_info:
                self.neuron_info['dt'] = t._timeseries.dt
            for k in t['data'].keys():
                new_names.add(k)
            # Check for metadata dictionary
            try:
                tm_dict = t[self.meta_dict_name]
            except KeyError:
                raise ValueError("Could not find meta data dictionary {0} for trial {1}.".format(self.meta_dict_name, t_ind))
            if not isinstance(tm_dict, dict):
                raise ValueError("Meta data for trial {0} is not formatted as a dictionary type.".format(t_ind))
            # Check consistency of global neuron info and update as needed
            for neuron_name in t['data'].keys():
                if neuron_name not in self.neuron_info.keys():
                    raise ValueError("Neuron name {0} found in trial {1} not recognized in neuron meta info!".format(neuron_name, t_ind))
                for meta_info in t[self.meta_dict_name][neuron_name].keys():
                    if meta_info.lower() == "class":
                        if self.neuron_info[neuron_name].cell_type != t[self.meta_dict_name][neuron_name][meta_info]:
                            raise ValueError("Neuron {0} has mismatched class ID {1} in trial {2} when {3} was expected.".format(neuron_name, t[self.meta_dict_name][neuron_name][meta_info], t_ind, self.neuron_info[neuron_name]['class']))
                        else:
                            # We found class in global and matches trial so this is good
                            pass
        for nn in new_names:
            if nn in self.__series_names.keys():
                raise ValueError("Session already has data name {0}.".format(nn))

        for nn in new_names:
            self.__series_names[nn] = data_type
        # Save dictionary reference to this trial set and conjoin to __main
        self._trial_lists[data_type] = ConjoinedList(trial_data)
        self._trial_lists['__main'].add_child(self._trial_lists[data_type])

        return None

    ##### METHODS THAT ARE SPECIFIC TO SESSIONS WITH NEURONS ADDED #########
    def gauss_convolved_FR(self, sigma, cutoff_sigma=4, series_name="_gauss",
                            set_as_default=True):
        """ Units of sigma must be given in ms and is converted to samples
        using the global neuron dt."""
        try:
            if not isinstance(self['neurons'], ConjoinedList):
                raise ValueError("Session must have the data_type 'neurons' added in a ConjoinedList to compute firing rates")
        except KeyError:
            raise ValueError("Session must have the data_type 'neurons' added in a ConjoinedList to compute firing rates")

        # x_win must be in units of bins
        sigma = sigma / self.neuron_info['dt']
        x_win = int(np.around(sigma * cutoff_sigma))
        xvals = np.arange(-1 * x_win, x_win + 1)
        kernel = np.exp(-.5 * (xvals / sigma) ** 2)
        kernel = kernel / np.sum(kernel)

        # Rescale from spike bin counts to firing rate
        fr_scale = 1000 / self.neuron_info['dt']
        new_names = set()
        # Loop over all neurons in each neuron trial
        for neuron_name in self.neuron_info['neuron_names']:
            new_series_name = neuron_name + series_name
            self.neuron_info['series_to_name'][new_series_name] = neuron_name
            for neuron_trial in self['neurons']:
                if neuron_trial[self.meta_dict_name][neuron_name]['spikes'] is None:
                    continue
                neuron_trial['data'][new_series_name] = fr_scale * np.convolve(neuron_trial['data'][neuron_name], kernel, mode='same')
                new_names.add(new_series_name)
            if set_as_default:
                # Assign this as default series for each neuron AFTER adding to session
                self.neuron_info[neuron_name].set_use_series(new_series_name)

        for nn in new_names:
            self.__series_names[nn] = "neurons"
        return None

    def set_default_neuron_series(self, series_name):
        """ Will search for neuron data series names containing the string in
        'series_name' and set that series to the default. If series_name is a
        list, this procedure is done for each element of the list.
        """
        if not isinstance(series_name, list):
            series_name = [series_name]
        for s_name in self.neuron_info['series_to_name']:
            for check_name in series_name:
                if check_name in s_names:
                    self.neuron_info[self.neuron_info[s_name]].set_use_series(s_name)

    def align_trial_data(self, alignment_event, alignment_offset=0.,
                         blocks=None, trial_sets=None):
        """ Aligns the trial timeseries to the event specified plus the
        offset specified.

        If trial_sets is None, the offset is applied to all trials containing
        the specified event.
        NOTE: since trial time is assumed to start at 0 in an absolute sense,
        alignement event times are subtracted from current alignment time after
        the offset is added to the event time. Thus, alignment to
        "fixation onset" with offset of 100 ms aligns to 100 ms after fixation
        onset = t0.
        """
        t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
        for ind in t_inds:
            st = self._session_trial_data[ind]
            if alignment_event not in st['events']:
                if trial_sets is not None:
                    print("WARNING: trial number {0} does not have an event named {1} and was not aligned.".format(ind, alignment_event))
                continue
            # Undo existing alignment and windows, resetting alignment to 0
            st['timeseries'] += st['aligned_time']
            st['curr_t_win']['time'] = [-np.inf, np.inf]
            # Set new alignment and metadata
            if st['events'][alignment_event] is None:
                # This trial has a field for this event but is not present
                # in this particular instance
                st['incl_align'] = False
            else:
                st['incl_align'] = True
                st['aligned_time'] = st['events'][alignment_event] + alignment_offset
                st['timeseries'] -= st['aligned_time']
        return None

    def shift_event_to_refresh(self, event_name, target_data_name='__main'):
        """ Finds each trial with an event 'event_name' and moves the event
        time to match the ms target sampling of the next refresh event (i.e.,
        shifts commanded event times to actual event times on screen).

        The alignment is peformed using the MaestroTarget object, which must
        be specified by name in 'target_data_name'."""
        for ind, st in enumerate(self._session_trial_data):
            try:
                st['events'][event_name] = self._trial_lists[target_data_name][ind]['data'].get_next_refresh(st['events'][event_name])
            except AttributeError:
                print("Data found in target_data_name {0} could not find MaestroTarget object in 'data' field.".format(target_data_name))
                continue
            except KeyError:
                # Trial does not have sought event so skip
                continue
            except:
                # Everything else also skip
                continue
        return None

    def get_data_list(self, series_name, time_window=None, blocks=None,
                      trial_sets=None):
        """ Returns a list of length trials, where each element of the list
        contains the timeseries data in the requested time window. If the time
        window exceeds the valid range of the timeseries, data are excluded.
        Thus the output list is NOT necessarily in 1-1 correspondence with the
        input trials sequence!
        Call "data_names()" to get a list of available data names. """
        data_out = []
        data_name = self.__series_names[series_name]
        if data_name == "neurons":
            # If requested data is a neuron we need to add its valid trial
            # index to the input trial_sets
            check_missing = True
            neuron_name = self.neuron_info['series_to_name'][series_name]
            trial_sets = self.neuron_info[neuron_name].append_valid_trial_set(trial_sets)
        else:
            check_missing = False
        t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
        for t in t_inds:
            if not self._session_trial_data[t]['incl_align']:
                # Trial is not aligned with others due to missing event
                continue
            trial_obj = self._trial_lists[data_name][t]
            if check_missing:
                if trial_obj[self.meta_dict_name][series_name]['spikes'] is None:
                    # Data are missing for this neuron trial series
                    continue
            trial_ts = trial_obj._timeseries
            try:
                if time_window is None:
                    # Use all indices
                    trial_tinds = trial_ts.indices()
                else:
                    trial_tinds = trial_ts.find_index_range(time_window[0], time_window[1])
            except IndexError:
                continue
            data_out.append(trial_obj['data'][series_name][trial_tinds])

        return data_out

    def get_data_array(self, series_name, time_window, blocks=None,
                        trial_sets=None, return_inds=False):
        """ Returns a n trials by m time points numpy array of the requested
        timeseries data. Missing data points are filled in with np.nan.
        Call "data_names()" to get a list of available data names. """
        data_out = []
        data_name = self.__series_names[series_name]
        if data_name == "neurons":
            # If requested data is a neuron we need to add its valid trial
            # index to the input trial_sets
            check_missing = True
            neuron_name = self.neuron_info['series_to_name'][series_name]
            trial_sets = self.neuron_info[neuron_name].append_valid_trial_set(trial_sets)
        else:
            check_missing = False
        t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
        t_inds_to_delete = []
        for i, t in enumerate(t_inds):
            if not self._session_trial_data[t]['incl_align']:
                # Trial is not aligned with others due to missing event
                t_inds_to_delete.append(i)
                continue
            trial_obj = self._trial_lists[data_name][t]
            if check_missing:
                if trial_obj[self.meta_dict_name][neuron_name]['spikes'] is None:
                    # Data are missing for this neuron trial series
                    t_inds_to_delete.append(i)
                    continue
            self._set_t_win(t, time_window)
            valid_tinds = self._session_trial_data[t]['curr_t_win']['valid_tinds']
            out_inds = self._session_trial_data[t]['curr_t_win']['out_inds']
            t_data = np.full(out_inds.shape[0], np.nan)
            t_data[out_inds] = trial_obj['data'][series_name][valid_tinds]
            data_out.append(t_data)

        data_out = [] if len(data_out) == 0 else np.vstack(data_out)
        if return_inds:
            if len(t_inds_to_delete) > 0:
                del_sel = np.zeros(t_inds.size, dtype='bool')
                del_sel[np.array(t_inds_to_delete, dtype=np.int64)] = True
                t_inds = t_inds[~del_sel]
            return data_out, t_inds
        else:
            return data_out

    def get_data_trial(self, trial_index, series_name, time_window):
        """ Returns a n trials by m time points numpy array of the requested
        timeseries data for ONE TRIAL. Missing data points are filled in with
        np.nan. Call "data_names()" to get a list of available data names. """
        data_out = []
        data_name = self.__series_names[series_name]
        trial_obj = self._trial_lists[data_name][trial_index]
        self._set_t_win(trial_index, time_window)
        valid_tinds = self._session_trial_data[trial_index]['curr_t_win']['valid_tinds']
        out_inds = self._session_trial_data[trial_index]['curr_t_win']['out_inds']
        if not self._session_trial_data[trial_index]['incl_align']:
            # Trial is not aligned due to missing event
            return np.zeros((0, out_inds.shape[0]))
        t_data = np.full(out_inds.shape[0], np.nan)
        t_data[out_inds] = trial_obj['data'][series_name][valid_tinds]
        return t_data

    def _get_trial_set(self, set_name):
        """Tries to get the trial indices associated with the specified trial
        set string and throws a more appropriate error during failure. If
        set_name is an appropriate numpy array trial set index, it is returned
        as-is. """
        if isinstance(set_name, np.ndarray):
            if len(set_name) != len(self):
                raise ValueError("Trial sets input as numpy ndarray must be the same size as the session, 1 entry per trial!")
            if set_name.dtype != 'bool':
                raise ValueError("Trial sets input as numpy ndarray must have boolean data type dtype=='bool'!")
            # Just return the input if it was a valid numpy array trial set
            return set_name
        try:
            t_set = self.trial_sets[set_name]
        except KeyError:
            raise KeyError("Session object has no trial set named {0}.".format(set_name))
        return t_set

    def _get_block(self, block_name):
        """Tries to get the trial indices associated with the specified block
        name string and throws a more appropriate error during failure. """
        try:
            t_block = self.blocks[block_name]
        except KeyError:
            raise KeyError("Session object has no block named {0}.".format(block_name))
        return t_block

    def __parse_block_to_indices(self, block):
        """Input is a block dictionary key or a raw Window object indicating
        the indices of a given block. Outputs the indices into the main trial
        list corresponding to the block. """
        if type(block) == Window:
            t_inds = np.arange(block[0], block[1], dtype=np.int32)
            return t_inds
        # block is not a Window object so try to find it in dictionary
        try:
            b_win = self._get_block(block)
            if b_win is None:
                return []
            t_inds = np.arange(b_win[0], b_win[1], dtype=np.int32)
            return t_inds
        except KeyError:
            raise ValueError("Block not found. Input block must be either a Window object or key to 'blocks' attribute dictionary.")

    def __parse_blocks_to_indices(self, blocks):
        """ Quick loop function that can iteratively parse multiple block
        strings in a list by calling __parse_block_to_indices. If blocks is
        input as a list of integers or numpy array of integers, these indices
        are returned as a unique, sorted numpy array of indices for trials. """
        is_raw_inds = False
        if isinstance(blocks, list):
            if len(blocks) == 0:
                # empty list
                return np.array([], dtype=np.int32)
            all_ints = True
            for el in blocks:
                if not isinstance(el, int):
                    # Failed our check that this is list of integer indices
                    all_ints = False
                    break
            if all_ints:
                is_raw_inds = True
                all_blk_indices = np.array(blocks, dtype=np.int32)
        if isinstance(blocks, np.ndarray):
            if len(blocks) == 0:
                # empty list
                return np.array([], dtype=np.int32)
            is_raw_inds = True
            all_blk_indices = np.int32(blocks)
        if is_raw_inds:
            # Above determined input was raw indices list or numpy array
            return np.unique(all_blk_indices) # Unique AND sorted

        if type(blocks) != list:
            blocks = [blocks]
        if len(blocks) == 0:
            # Blocks is empty
            return np.array([], dtype=np.int32)
        all_blk_indices = []
        for blk in blocks:
            all_blk_indices.append(self.__parse_block_to_indices(blk))
        if len(all_blk_indices) > 0:
            # Can only do this if indices are found
            all_blk_indices = np.int32(np.hstack(all_blk_indices))
            all_blk_indices = np.unique(all_blk_indices) # Unique AND sorted
        else:
            # Still return numpy indices
            all_blk_indices = np.array([], dtype=np.int32)
        return all_blk_indices

    def __blocks_and_trials_to_indices(self, blocks, trial_sets):
        """ Given a key, or list of keys, to the blocks and trial sets dictionaries
        this function returns the indices of trials satisfiying the condition:
        --T in at least 1 block in blocks and at least 1 set in trial_sets--
        where T is a trial whose index is included in the final output. """
        # Gather all the possible block indices indicated by blocks
        if blocks is None:
            all_blk_indices = np.arange(0, len(self), dtype=np.int32)
        else:
            all_blk_indices = self.__parse_blocks_to_indices(blocks)
        # Gather all the possible trial_sets indicated by trial_sets
        if type(trial_sets) != list:
            trial_sets = [trial_sets]
        all_trial_sets = self._get_trial_set(trial_sets[0])
        if len(trial_sets) > 1:
            for ts in trial_sets[1:]:
                all_trial_sets = np.logical_and(all_trial_sets, self._get_trial_set(ts))
        # Scan all block indices and trials
        keep_inds = np.zeros(all_blk_indices.shape[0], dtype='bool')
        for n_ind, t_ind in enumerate(all_blk_indices):
            if all_trial_sets[t_ind]:
                keep_inds[n_ind] = True
        all_blk_indices = all_blk_indices[keep_inds]
        return all_blk_indices

    def _parse_blocks_trial_sets(self, blocks=None, trial_sets=None):
        """ Primarily parses inputs of None to indicate all trials and
        otherwise calls __blocks_and_trials_to_indices() above for main work.
        """
        if blocks is None and trial_sets is None:
            return np.arange(0, len(self), dtype=np.int32)
        elif blocks is not None and trial_sets is None:
            return self.__parse_blocks_to_indices(blocks)
        else:
            return self.__blocks_and_trials_to_indices(blocks, trial_sets)

    def _parse_time_to_indices(self, time):
        """ Accepts a 2 element time window, slice of times or array/list of
        times and turns them into the corresponding indices using the
        timeseries of the trial. """
        pass

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
                raise ValueError("New trial data in trial number {0} trial name does not match existing session trial name ({1} vs. {2}).".format(t_ind, t['name'], self._session_trial_data[t_ind]['name']))
            if len(t._timeseries) != len(self._session_trial_data[t_ind]['timeseries']):
                raise ValueError("Length of new trial data in trial number {0} does not match existing session trial.".format(t_ind))
            if t._timeseries[0] != self._session_trial_data[t_ind]['timeseries'][0]:
                raise ValueError("New trial data in trial number {0} is not aligned to the same time for index zero as existing session trial ({1} vs. {2}).".format(t_ind, t._timeseries[0], self._session_trial_data[t_ind][0]))

        return None

    def add_blocks(self, trial_names, block_names, number_names=False, **criteria):
        """ Adds groups of trials to blocks named 'block_names'. Blocks are
        named in order of trial number using the order of names provided in
        block_names. If more blocks are found than block_names given, and error
        is raised. If fewer blocks are found than names given, the extra names
        are ignored.
        """
        trial_windows, _ = find_trial_blocks(self._trial_lists['__main'], trial_names, **criteria)
        if number_names:
            if isinstance(block_names, str):
                block_names = [block_names]
            new_names = []
            for tw_ind in range(0, len(trial_windows)):
                if tw_ind < len(block_names):
                    new_names.append(block_names[tw_ind])
                else:
                    if tw_ind > 99:
                        raise ValueError("Not ready to handle over 99 newly numbered blocks with only 2 digits!")
                    digit = f"{tw_ind:02}"
                    new_names.append(block_names[0] + "_num" + digit)
            block_names = new_names

        if type(block_names) == list:
            if len(trial_windows) > len(block_names):
                raise RuntimeError("Found more blocks than names given. Add block_names or check that criteria are appropriately strict.")
            for tw_ind in range(0, len(trial_windows)):
                self.blocks[block_names[tw_ind]] = trial_windows[tw_ind]
        else:
            if len(trial_windows) > 1:
                raise RuntimeError("Found more blocks than names given. Add block_names or check that criteria are appropriately strict.")
            else:
                self.blocks[block_name] = trial_windows[0]
        return None

    def add_trial_set(self, new_set_name, trials=None, blocks=None):
        """ Finds the trials satisfying the "trials" criterion that are within
        the blocks given by "blocks". Stores the result under the key
        "new_set_name" in the dictionary self.trial_sets as a boolean index
        of trials satisfying the criteria.
        Slice inputs for "trials" will be taken with respect to the input blocks,
        for example trials=slice(0:10) will return the first 10 trials starting
        from the first index of "blocks". If trials is a string, it is converted
        to a list. All list inputs for trials are assumed to be a list of
        strings of trial names. To scan specific indices for trials, the
        indices must be input in a numpy array rather than a list and will
        be selected with respect to the input blocks as done for slices.
        """
        new_set = np.zeros(len(self), dtype='bool')
        if trials is None and blocks is None:
            # No blocks or trials specified so add everything
            new_set[:] = True
            self.trial_sets[new_set_name] = new_set
            return None
        # Making it here means at least 1 of trials or blocks is NOT None.
        # Find the indices corresponding to the input blocks.
        if blocks is None:
            t_inds = np.arange(0, len(self))
        else:
            t_inds = self.__parse_blocks_to_indices(self, blocks)
        if trials is None:
            # No trials specified, so add the whole blocks and we're done
            new_set[t_inds] = True
            self.trial_sets[new_set_name] = new_set
            return None
        # If we made it here, trials is NOT None so must parse it.
        if type(trials) == str:
            # Put string in a list to iterate
            trials = [trials]
        if type(trials) == list:
            # Check list is list of strings
            for t_name in trials:
                if type(t_name) != str:
                    raise ValueError("Inputs must be strings of trial names. To input trial indices use a numpy array of integers or python slice object.")
            # Search for trials with this name in the input blocks
            for ind in t_inds:
                if self._session_trial_data[ind]['name'] in trials:
                    new_set[ind] = True
        elif type(trials) == slice:
            # Return the slice of trial indices
            t_inds = t_inds[trials]
            new_set[t_inds] = True
        elif type(trials) == np.ndarray:
            if trials.dtype == 'bool':
                if trials.shape[0] == len(self):
                    new_set = trials
                elif trials.shape[0] == t_inds.shape[0]:
                    t_inds = t_inds[trials]
                    new_set[t_inds] = True
                else:
                    raise ValueError("Assignment of numpy boolean trial set must be an array equal to len(self) ({0}) or len(blocks) ({1}) input.".format(len(self), t_inds.shape[0]))
            else:
                # Return the trial indices input for this blocks
                t_inds = t_inds[trials]
                new_set[t_inds] = True
        else:
            raise ValueError("Invalid type used to specify trials. Must be type 'None', 'str' (or list of str), 'slice' or 'numpy.ndarray'.")
        self.trial_sets[new_set_name] = new_set
        return None

    def find_trials_less_than_event(self, event, blocks=None,
                    trial_sets=None, ignore_not_found=True, event_offset=0.):
        """Returns an index of trials whose duration is less than the input
        event time. This will be done only within the input trial and block
        names (if any). Output can then be put into the "delete_trials"
        function which must handle updating any blocks, trial sets, indices,
        etc. that will be affected since the super class cannot know these.
        Trials that lack the event name specified in 'event' are marked as
        being less than the event if ignore_not_found is 'False'. Events that
        return a time of 'None' are treated as being less than event. """
        trials_less_than_event = np.zeros(len(self), dtype='bool')
        t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
        for ind in t_inds:
            # Try assuming that event is in events dictionary
            try:
                if self[ind].events[event] is None:
                    trials_less_than_event[ind] = True
                elif self[ind].duration < (self[ind].events[event] + event_offset):
                    trials_less_than_event[ind] = True
                else:
                    # This should imply that the trial has the event
                    # Here is just setting to default value for clarity, but
                    # is not necessary
                    trials_less_than_event[ind] = False
            except KeyError:
                # Trial does not have this event
                if ignore_not_found:
                    trials_less_than_event[ind] = False
                else:
                    trials_less_than_event[ind] = True
            except:
                print("ind: ", ind, "duration: ", self[ind].duration, "event time: ", self[ind].events[event])
                raise
        return trials_less_than_event

    def delete_trials(self, indices):
        """ Deletes the set of trials indicated by indices. Indices are either
        integer indices of the trials to remove or a boolean mask where the
        "True" values will be deleted. """
        if type(indices) == np.ndarray:
            if indices.dtype == 'bool':
                d_inds = np.int64(np.nonzero(indices)[0])
            else:
                # Only unique, sorted, as integers
                d_inds = np.int64(indices)
        elif type(indices) == list:
            d_inds = np.array(indices, dtype=np.int64)
        else:
            if type(indices) != int:
                raise ValueError("Indices must be a numpy array, python list, or integer type.")
        if d_inds.ndim > 1:
            raise ValueError("Indices must be 1 dimensional or scalar.")
        if (type(indices) == int) or (d_inds.ndim == 0):
            # Only 1 index input for deletion
            del self[indices]
            return None
        if d_inds.shape[0] == 0:
            # Nothing to delete
            return None
        # Multiple indices are present. Use unique, sorted, in reverse
        d_inds = np.unique(d_inds)[-1::-1]
        # Use del function to update blocks and trials in reverse order
        for ind in d_inds:
            # To avoid constantly re-allocating numpy arrays, we can delete all of these at the end
            self.__del_trial_set__ = False
            del self[ind]
        # Now remove trial sets all at once
        for ts in self.trial_sets.keys():
            self.trial_sets[ts] = np.delete(self.trial_sets[ts], d_inds)
        self.__verify_data_lengths()
        return None

    def alias_trial_name(self, trials, old_name, new_name):
        pass


    ########## A SET OF NAME DISPLAY FUNCTIONS FOR USEFUL PROPERTIES ##########
    def get_trial_names(self):
        trial_names = []
        for trial in self.trials:
            if trial['trial_name'] not in trial_names:
                trial_names.append(trial['trial_name'])
        return trial_names

    def get_data_names(self):
        """Provides a list of the available data names. """
        return [x for x in self._trial_lists.keys() if x[0:2] != "__"]

    def get_series_names(self):
        """Provides a list of the available data series names under the given
        data_name. """
        return [x for x in self.__series_names.keys()]

    def get_event_names(self):
        """ Provides a list of all event names in the entire data set. """
        all_names = set()
        for st in self._session_trial_data:
            for ev in st['events']:
                all_names.add(ev)
        return [x for x in all_names]

    def get_trial_set_names(self):
        """ Provides a list of all trial_set names in the session object. """
        all_names = set()
        for ts in self.trial_sets.keys():
            all_names.add(ts)
        return [x for x in all_names]

    def get_block_names(self):
        """ Provides a list of all block names in the session object. """
        all_names = set()
        for blk in self.blocks.keys():
            all_names.add(blk)
        return [x for x in all_names]

    def get_neuron_names(self):
        """ Provides a list of all neuron names joined to the session object. """
        all_names = set()
        for n_name in self.neuron_info['neuron_names']:
            all_names.add(n_name)
        return [x for x in all_names]
    ###########################################################################


    def __verify_data_lengths(self):
        for blk in self.blocks.keys():
            blk_win = self.blocks[blk]
            if blk_win is None:
                # Don't check empty blocks
                continue
            if ( (blk_win[0] < 0) or (blk_win[0] > len(self))
                or (blk_win[1] < 1) or (blk_win[1] > len(self)) ):
                raise RuntimeError("The block window for block {0} has moved beyond the range of viable trials. Likely the result of error during trial deletion.".format(blk))
        for ts in self.trial_sets.keys():
            if type(self.trial_sets[ts]) != np.ndarray:
                raise RuntimeError("The trial set '{0}' is not of type numpy.ndarray.".format(ts))
            if self.trial_sets[ts].shape[0] != len(self):
                raise RuntimeError("The trial set '{0}' is not the same length as the trials in session. Likely the result of error during trial deletion.".format(ts))
        return None

    def __delitem__(self, index):
        """ Deletes the trial and info associated with index. This can be tricky
        but the goal here is to keep track of everything so trials are in 1-1
        alignment. Doing more than checking lengths would be hard here, but this
        is already managed in ConjoinedList so here just track other stuff.
        Currently removes:
            blocks
            _trial_lists
            trial_sets
        """
        # Update the block windows
        checked_wins = set()
        for blk in self.blocks:
            blk_win = self.blocks[blk]
            if blk_win is None:
                # Block doesn't exist so skip
                continue
            if blk_win in checked_wins:
                # Block windows can be aliased so don't duplicate
                continue
            if blk_win[0] > index:
                blk_win[0] -= 1
                blk_win[1] -= 1
            elif (blk_win[0] < index) and (blk_win[1] > index):
                blk_win[1] -= 1
            elif blk_win[0] == index:
                blk_win[1] -= 1
            # elif blk_win[1] == index:
            #     blk_win[1] -= 1
            elif blk_win[1] <= index:
                pass
            else:
                raise ValueError("Index to window contingency not found for block window '{0}' and index '{1}'!".format(blk_win, index))
            checked_wins.add(blk_win)
        # Deleting main part of conjoined list should delete everything else
        del self._trial_lists['__main'][index]
        # Must update trial sets according to what was deleted
        if self.__del_trial_set__:
            for ts in self.trial_sets.keys():
                self.trial_sets[ts] = np.delete(self.trial_sets[ts], index)
            self.__verify_data_lengths()
        # If neurons are present, we need to recompute their tuning/fits on the
        # newly updated trial sets
        if hasattr(self, "neuron_info"):
            if isinstance(self.neuron_info, dict):
                for n_name in self.neuron_info['neuron_names']:
                    self.neuron_info[n_name].recompute_fits()
        self.__del_trial_set__ = True # Always set back to True so this function works as normal
        return None

    def __len__(self):
        return len(self._trial_lists['__main'])

    def __getitem__(self, item):
        """ Items are retrieved by first assuming integer/slice index into the
        trial lists but then checks for strings to dictionary/attributes if
        an error is raised. """
        try:
            # Just assume integer index
            return self._trial_lists['__main'][item]
        except TypeError:
            if type(item) == str:
                try:
                    return self._trial_lists[item]
                except KeyError:
                    raise ValueError("Could not find data name '{0}'. Use method 'data_names()' for list of valid data names".format(item))
            else:
                raise ValueError("Cannot find item or trial index '{0}' in session".format(item))
