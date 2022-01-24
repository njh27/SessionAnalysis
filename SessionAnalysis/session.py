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


class Session(list):
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
    trial_data : python list of Trial type objects
        Each element of the list must contain an object of the Trial class
        specified in the trial.py module. This is not explicitly checked
        because double imports can confuse checking so undefined errors could
        result if other types are used. The first list of trial objects will
        be treated as the parent for any future trial sets added.
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

    def __init__(self, trial_data, session_name=None, data_type=None):
        """
        """
        self._trial_lists = {}
        self._trial_lists['__main'] = ConjoinedList(trial_data)
        if data_type is None:
            # Use data name of first trial as default
            data_type = trial_data[0].__data_alias__
        self.__data_fields = [data_type]
        # The first data listed is hard coded here as the parent of conjoined lists
        self._trial_lists[data_type] = self._trial_lists['__main']
        self.__data_names = {}
        for t in trial_data:
            for k in t['data'].keys():
                self.__data_names[k] = data_type
        self.session_name = session_name
        self.blocks = {}
        self.neurons = []

    def add_trial_data(self, trial_data, data_type=None):
        if data_type is None:
            # Use data name of first trial as default
            data_type = trial_data[0].__data_alias__
        if data_type in self.__data_fields:
            raise ValueError("Session already has data type {0}.".format(data_type))
        # Check to update data_names so we can find their associated lis of
        # trials quickly later (e.g. in get_data)
        new_names = set()
        for t in trial_data:
            for k in t['data'].keys():
                new_names.add(k)
        for nn in new_names:
            if nn in self.__data_names.keys():
                raise ValueError("Session already has data name {0}.".format(nn))

        self.__data_fields.append(data_type)
        for nn in new_names:
            self.__data_names[nn] = data_type
        # Save dictionary reference to this trial set and conjoin to __main
        self._trial_lists[data_type] = ConjoinedList(trial_data)
        self._trial_lists['__main'].add_child(self._trial_lists[data_type])

        return None

    def get_data(self, data_name, series_name, trials, time):

        data_out = []
        try:
            d_type = self.__data_names[data_name]
        except KeyError:
            raise KeyError("Session does not have a trial dataset with data name {0}.".format(data_name))

        for t in trials:
            trial_obj = self._trial_lists[d_type][t]
            trial_ts = trial_obj._timeseries
            trial_tinds = trial_ts.find_index_range(time[0], time[1])
            # data_out.append(trial_obj[data_name][series_name][trial_tinds])
            data_out.append(trial_tinds)

        return data_out

    def __parse_trials_to_indices(self, trials):
        """ Can accept string inputs indicating block names, or slices of indices,
        or numpy array of indices, or list of indices and outputs a corresponding
        useful index for getting the corresponding trials."""
        pass

    def __parse_time_to_indices(self, time):
        """ Accepts a 2 element time window, slice of times or array/list of
        times and turns them into the corresponding indices using the
        timeseries of the trial. """
        pass




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

    def __len__(self):
        return len(self._trial_lists['__main'])

    def get(self, key):
        return self.__getitem__(key)

    def setdefault(self, key, default=None):
        try:
            return self.get(key)
        except:
            return default

    def keys(self):
        return self.__data_fields

    def __getitem__(self, key):
        try:
            return self._trial_lists[key]
        except AttributeError:
            for k in self.data.keys():
                if k == key:
                    return self.data[k]
            for k in self.events.keys():
                if k == key:
                    return self.events[k]
            raise ValueError("Could not find value '{0}'.".format(key))
        except TypeError:
            raise TypeError("Quick references to attributes of Trial objects must be string.")
