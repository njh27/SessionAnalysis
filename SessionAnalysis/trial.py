""" This module defines a generic "Trial" type from which the "apparatus",
"neuron", and "behavior" trial types inherit. Objects of this class are
fed into the Session class for indexing and analysis. """



class Trial(dict):
    """ A class containing a generic trial and its associated timeseries and
    events. Data are organized and class functions are definted to make it
    easier to index and analyze associated data.

    Parameters
    ----------
    trial_dict : python dict
        The dictionary must specifiy the following trial attributes via its
        keys to define a Trial. These keys are NOT CASE SENSITIVE. These keys
        will be mapped to lower case versions. Any additional keys will remain
        exactly as input. Integer keys are NOT allowed as the trial data can be
        indexed by integers and slices. Recommended not to use duplicate keys
        within the data and events dicts, but in case they are, keys for the
        data dict are searched first. Any additional keys are not checked by
        the builtin/inherited functions (e.g. __getitem__, __delitem___ etc.)
        name : str
            Name of the given trial.
        data : dict
            Dictionary containing time dependent data such as stimulus and/or
            behavioral movement data. Each key of the dictionary specifies a
            timeseries. The key names for this dictionary can be used later to
            reference and extract their corresponding data.
        events : dict
            Dictionary where each key names an event. Each event is a time
            within the session on which the timeseries data (e.g. neural and
            behavioral data) can be aligned. Event times should therefor be in
            agreement with the time scheme of the timeseries data, i.e. starting
            the beginning of each trial t=0 or relative to the entire session.
            Event names can be used to reference events for alignment.

    Returns
    -------
    object of Trial class
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

    def get(self, key):
        return self.__getitem__(key)

    def setdefault(self, key, default=None):
        try:
            return self.get(key)
        except:
            return default

    def keys(self):
        return [x for x in self]

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
            raise ValueError("Could not resolve reqested index {0}.".format(index))

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

    def __update__(self):
        print("updating")

    def __setitem__(self, key, value):
        if type(key) == int:
            raise ValueError("'Trial' objects can not use integers as keys/attributes.")
        setattr(self, key, value)
        return None

    def update(self, new_data):
        if type(new_data) == dict:
            for key, value in new_data.items():
                self.__setitem__(key, value)
            return None
        # If not dictionary try assuming it's an iterable of pairs
        try:
            for key, value in new_data:
                self.__setitem__(key, value)
            return None
        except:
            raise ValueError("Inputs must be a dictionary or iterable of 'key', 'value' pairs.")

    """ Below are overrriden to be not defined from dictionary class"""
    def clear(self):
        raise AttributeError("'Trial' object has no attribute '{0}'.".format("clear"))
    def copy(self):
        raise AttributeError("'Trial' object has no attribute '{0}'.".format("copy"))
    def items(self):
        raise AttributeError("'Trial' object has no attribute '{0}'.".format("items"))
    def pop(self):
        raise AttributeError("'Trial' object has no attribute '{0}'.".format("pop"))
    def popitem(self):
        raise AttributeError("'Trial' object has no attribute '{0}'.".format("popitem"))
    def values(self):
        raise AttributeError("'Trial' object has no attribute '{0}'.".format("values"))


class ApparatusTrial(Trial):
    """ A class containing the timeseries associated with a trial experiemental
    apparatus, e.g. target motion. See Trial class.

    Parameters
    ----------
    trial_dict : python dict
        The dictionary must specifiy the following trial attributes via its
        keys to define a Trial. These keys are NOT CASE SENSITIVE. These keys
        will be mapped to lower case versions. Any additional keys will remain
        exactly as input. Integer keys are NOT allowed as the trial data can be
        indexed by integers and slices. Recommended not to use duplicate keys
        within the data and events dicts, but in case they are, keys for the
        data dict are searched first. Any additional keys are not checked by
        the builtin/inherited functions (e.g. __getitem__, __delitem___ etc.)
        name : str
            Name of the given trial.
        data : dict
            Dictionary containing time dependent data such as stimulus and/or
            behavioral movement data. Each key of the dictionary specifies a
            timeseries. The key names for this dictionary can be used later to
            reference and extract their corresponding data.
        events : dict
            Dictionary where each key names an event. Each event is a time
            within the session on which the timeseries data (e.g. neural and
            behavioral data) can be aligned. Event times should therefor be in
            agreement with the time scheme of the timeseries data, i.e. starting
            the beginning of each trial t=0 or relative to the entire session.
            Event names can be used to reference events for alignment.
    data_name : str
        Name used to describe the type of data and provide easy access for
        indexing the data in trial_dict[data]. e.g. "target"

    Returns
    -------
    object of ApparatusTrial class
    """

    def __init__(self, trial_dict, data_name="apparatus"):
        """
        """
        if type(data_name) != str:
            raise ValueError("ApparatusTrial object must have a string for data_name.")
        self.__data_alias__ = data_name
        Trial.__init__(self, trial_dict)

    def __getitem__(self, index):
        # Use data alias as shortcut to indexing data
        if self.__data_alias__ in index:
            if type(index) == tuple:
                # Multiple attribute/indices input so split
                attribute = index[0]
                index = index[1:]
                if len(index) == 1: index = index[0]
            else:
                attribute = index
                index = None
            if index is None:
                return self.data
            else:
                return self.data[index]
        else:
            return Trial.__getitem__(self, index)

    def __build_iterator__(self):
        iter_str = [x for x in self.__dict__.keys()]
        for x in ["data", "events"]:
            for key in self[x]:
                iter_str.append(key)
        # Modify to substitute data_name/alias for "data"
        iter_str.remove("data")
        iter_str.append(self.__data_alias__)
        self.__myiterator__ = iter(iter_str)


class BehavioralTrial(Trial):
    """ A class containing the timeseries associated with a trial's behavioral
    data, e.g. eye position. See Trial class.

    Parameters
    ----------
    trial_dict : python dict
        The dictionary must specifiy the following trial attributes via its
        keys to define a Trial. These keys are NOT CASE SENSITIVE. These keys
        will be mapped to lower case versions. Any additional keys will remain
        exactly as input. Integer keys are NOT allowed as the trial data can be
        indexed by integers and slices. Recommended not to use duplicate keys
        within the data and events dicts, but in case they are, keys for the
        data dict are searched first. Any additional keys are not checked by
        the builtin/inherited functions (e.g. __getitem__, __delitem___ etc.)
        name : str
            Name of the given trial.
        data : dict
            Dictionary containing time dependent data such as stimulus and/or
            behavioral movement data. Each key of the dictionary specifies a
            timeseries. The key names for this dictionary can be used later to
            reference and extract their corresponding data.
        events : dict
            Dictionary where each key names an event. Each event is a time
            within the session on which the timeseries data (e.g. neural and
            behavioral data) can be aligned. Event times should therefor be in
            agreement with the time scheme of the timeseries data, i.e. starting
            the beginning of each trial t=0 or relative to the entire session.
            Event names can be used to reference events for alignment.
    data_name : str
        Name used to describe the type of data and provide easy access for
        indexing the data in trial_dict[data]. e.g. "eye_pos"

    Returns
    -------
    object of BehavioralTrial class
    """

    def __init__(self, trial_dict, data_name="behavior"):
        """
        """
        if type(data_name) != str:
            raise ValueError("BehavioralTrial object must have a string for data_name.")
        self.__data_alias__ = data_name
        Trial.__init__(self, trial_dict)

    def __getitem__(self, index):
        # Use data alias as shortcut to indexing data
        if self.__data_alias__ in index:
            if type(index) == tuple:
                # Multiple attribute/indices input so split
                attribute = index[0]
                index = index[1:]
                if len(index) == 1: index = index[0]
            else:
                attribute = index
                index = None
            if index is None:
                return self.data
            else:
                return self.data[index]
        else:
            return Trial.__getitem__(self, index)

    def __build_iterator__(self):
        iter_str = [x for x in self.__dict__.keys()]
        for x in ["data", "events"]:
            for key in self[x]:
                iter_str.append(key)
        # Modify to substitute data_name/alias for "data"
        iter_str.remove("data")
        iter_str.append(self.__data_alias__)
        self.__myiterator__ = iter(iter_str)


class NeuronTrial(Trial):
    """ A class containing the timeseries associated with a trial for a single
    neuron's responses. See Trial class.

    Parameters
    ----------
    trial_dict : python dict
        The dictionary must specifiy the following trial attributes via its
        keys to define a Trial. These keys are NOT CASE SENSITIVE. These keys
        will be mapped to lower case versions. Any additional keys will remain
        exactly as input. Integer keys are NOT allowed as the trial data can be
        indexed by integers and slices. Recommended not to use duplicate keys
        within the data and events dicts, but in case they are, keys for the
        data dict are searched first. Any additional keys are not checked by
        the builtin/inherited functions (e.g. __getitem__, __delitem___ etc.)
        name : str
            Name of the given trial.
        data : dict
            Dictionary containing time dependent data such as stimulus and/or
            behavioral movement data. Each key of the dictionary specifies a
            timeseries. The key names for this dictionary can be used later to
            reference and extract their corresponding data.
        events : dict
            Dictionary where each key names an event. Each event is a time
            within the session on which the timeseries data (e.g. neural and
            behavioral data) can be aligned. Event times should therefor be in
            agreement with the time scheme of the timeseries data, i.e. starting
            the beginning of each trial t=0 or relative to the entire session.
            Event names can be used to reference events for alignment.
    data_name : str
        Name used to describe the type of data and provide easy access for
        indexing the data in trial_dict[data]. e.g. "spikes"
    classif : str
        Name used to describe the classification or type of neuron in this
        trial. This name can be used in later analyses to filter specific
        neurons.

    Returns
    -------
    object of NeuronTrial class
    """

    def __init__(self, trial_dict, data_name="spikes", classif=None):
        """
        """
        if type(data_name) != str:
            raise ValueError("NeuronTrial object must have a string for data_name.")
        self.__data_alias__ = data_name
        self.classif = classif
        Trial.__init__(self, trial_dict)

    def __getitem__(self, index):
        # Use data alias as shortcut to indexing data
        if self.__data_alias__ in index:
            if type(index) == tuple:
                # Multiple attribute/indices input so split
                attribute = index[0]
                index = index[1:]
                if len(index) == 1: index = index[0]
            else:
                attribute = index
                index = None
            if index is None:
                return self.data
            else:
                return self.data[index]
        else:
            return Trial.__getitem__(self, index)

    def __build_iterator__(self):
        iter_str = [x for x in self.__dict__.keys()]
        for x in ["data", "events"]:
            for key in self[x]:
                iter_str.append(key)
        # Modify to substitute data_name/alias for "data"
        iter_str.remove("data")
        iter_str.append(self.__data_alias__)
        self.__myiterator__ = iter(iter_str)
