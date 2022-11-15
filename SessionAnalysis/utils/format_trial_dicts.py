import numpy as np
from SessionAnalysis.trial import ApparatusTrial, BehavioralTrial, NeuronTrial
from SessionAnalysis.utils.general import Indexer



def is_compressed(t):
    try:
        _ = t['compressed_target']
        return True
    except KeyError:
        return False


def events_list_to_dict(events_list, event_names=None, missing_event=None,
                        convert_to_ms=True):
    """ Assigns the list of events of a single Maestro trial to a dictionary.

    If event names are given, only these events are kept in the output. If
    event names is None, all events are named numerically as:
        event_dict[DIO channel][pulse ordered index]
    Here the key "DIO channel" will be a STR type! The ordered index will be
    an int.

    Parameters
    ----------
    events_list : python list
        The list of events as output in MaestroRead (maestro_data['events']
        with each element being a list corresponding to the pulses of a DIO
        channel. Each internal list value contains the timestamp for each pulse
        on the DIO channel. If this is input as a dictionary, it is assumed
        events have already been correctly formatted and the input is returned.
    event_names : python dict
        The dictionary contains desired names for events as its keys. Each key
        must return a two element tuple or list of integers that specify the
        DIO channel and ordered event number on that channel corresponding to
        the name of the desired event. Keys MUST be type "str" to be compatible
        with Trial objects.
    missing_event : str("skip") or None
        Determine what to do with trials that do not have the requested event
        (e.g. incomplete trials). If "skip", nothing is done and the new events
        dictionary will not have a key for the specified event. If None, or
        any other value, then that value is assigned to the key for the
        specified event.
    convert_to_ms : bool
        If True, all event times are multiplied by 1000 to convert to ms (the
        default Maestro output is in seconds).

    Returns
    -------
    event_dict : python dict
        The dictionary returns the time of events as assigned to the keys
        given in the input event_names dictionary.

    Examples
    -------
            event_names["target_onset"] = [4, 2]
        Indicates that the 3rd event (0 based indexing => 2 is 3rd) on DIO
        channel 4 (also zero based indexing as in Maestro) indicates target
        onset. The output event_dict["target_onset"] will give the timestamp
        corresponding to this event.
    """
    if type(events_list) == dict:
        return events_list
    event_dict = {}
    if event_names is None:
        for dio_ind, dio_chan in enumerate(events_list):
            if len(dio_chan) > 0:
                dio_key = str(dio_ind)
                event_dict[dio_key] = {}
                for evn_ind, evn in enumerate(dio_chan):
                    if convert_to_ms:
                        evn = evn * 1000
                    event_dict[dio_key][evn_ind] = evn
    else:
        for evn in event_names.keys():
            if type(evn) != str:
                raise ValueError("Keys for event names must be type string, not '{0}' as found for key '{1}'".format(type(evn), evn))
            try:
                event_dict[evn] = events_list[event_names[evn][0]][event_names[evn][1]]
                if convert_to_ms:
                    event_dict[evn] = event_dict[evn] * 1000
            except IndexError:
                # The requested index does not exist for this event name
                if missing_event is None:
                    event_dict[evn] = missing_event
                elif missing_event.lower() == "skip":
                    continue
                else:
                    event_dict[evn] = missing_event

    return event_dict


def format_maestro_events(maestro_data, event_names_by_trial=None,
                          missing_event=None, convert_to_ms=True):
    """ Reformats events data for each Maestro trial IN PLACE according to the
    dictionary requirements of Trial objects by iteratively calling
    events_list_to_dict.

    If event names are given, only these events are kept in the output. If
    event names is None, all events are named numerically (see
    "events_list_to_dict" above for more info).

    Parameters
    ----------
    events_list : python list
        The list of events as output in MaestroRead (maestro_data['events']
        with each element being a list corresponding to the pulses of a DIO
        channel. Each internal list value contains the timestamp for each pulse
        on the DIO channel. If this is input as a dictionary, it is assumed
        events have already been correctly formatted and the input is returned.
    event_names_by_trial : python dict
        A dictionary who's keys are the trial names. Each trial name key
        returns the desired event names for that trial type. The event names
        should be formatted as described in "events_list_to_dict" as they will
        be passed directly into this function.
    missing_event : str("skip") or None
        Determine what to do with trials that do not have the requested event
        (e.g. incomplete trials). If "skip", nothing is done and the new events
        dictionary will not have a key for the specified event. If None, or
        any other value, then that value is assigned to the key for the
        specified event.
    convert_to_ms : bool
        If True, all event times are multiplied by 1000 to convert to ms (the
        default Maestro output is in seconds).

    Returns
    -------
    None : python None
        The input list of Maestro trials maestro_data is modified in place.
    """

    for t in maestro_data:
        if type(t['events']) == dict:
            continue
        try:
            event_names = event_names_by_trial[t['header']['name']]
        except KeyError:
            event_names = None
        # Preserve original event format for neuron alignment
        t['og_events'] = t['events']
        t['events'] = events_list_to_dict(t['events'], event_names,
                        missing_event, convert_to_ms)

    return None


def maestro_to_trial(maestro_data):
    """ Convert maestro_data to format suitable for making Trial objects.
    NOTE: Not clear this is useful at the moment, when specific conversion
    functions below for apparatus and behavior should be used.
    """
    keep_keys = ['horizontal_eye_position',
                 'vertical_eye_position',
                 'horizontal_eye_velocity',
                 'vertical_eye_velocity']

    optn_keys = ['horizontal_target_position',
                 'vertical_target_position',
                 'horizontal_target_velocity',
                 'vertical_target_velocity']
    trial_list = []
    for t in maestro_data:
        tdict = {}
        tdict['name'] = t['header']['name']
        tdict['events'] = t['events']
        tdict['data'] = {}
        if is_compressed(t):
            for key in keep_keys:
                tdict['data'][key] = t[key]
            tdict['data']['targets'] = t['targets']
        else:
            for key in keep_keys:
                tdict['data'][key] = t[key]
            for key in optn_keys:
                tdict['data'][key] = t[key]
        trial_list.append(tdict)

    return trial_list


def maestro_to_apparatus_trial(maestro_data, target_num, dt_data, start_data=0,
                                data_name="apparatus"):
    """ Convert maestro_data to format suitable for making ApparatusTrial
    objects for a single target object indexed by "target_num". """
    optn_keys = ['horizontal_target_position',
                 'vertical_target_position',
                 'horizontal_target_velocity',
                 'vertical_target_velocity']
    trial_list = []
    for t in maestro_data:
        tdict = {}
        tdict['name'] = t['header']['name']
        tdict['used_stab'] = t['header']['UsedStab']
        # Need to format events list to events dictionary
        tdict['events'] = t['events']
        tdict['data'] = {}
        if is_compressed(t):
            # Need to format list of targets to dictionary of 1 target
            tdict['data'] = t['targets'][target_num]
        else:
            for key in optn_keys:
                tdict['data'][key] = t[key]
        trial_list.append(ApparatusTrial(tdict, dt_data, start_data, data_name))

    return trial_list


def maestro_to_behavior_trial(maestro_data, dt_data, start_data=0,
                                data_name="movement"):
    """ Convert maestro_data to format suitable for making BehaviorTrial
    objects. """
    keep_keys = ['horizontal_eye_position',
                 'vertical_eye_position',
                 'horizontal_eye_velocity',
                 'vertical_eye_velocity']
    trial_list = []
    for t in maestro_data:
        tdict = {}
        tdict['name'] = t['header']['name']
        tdict['events'] = t['events']
        tdict['data'] = {}
        for key in keep_keys:
            tdict['data'][key] = t[key]
        trial_list.append(BehavioralTrial(tdict, dt_data, start_data, data_name))

    return trial_list


def maestro_to_neuron_trial(maestro_data, neurons, dt_data=None, start_data=0,
                            default_name="n_", use_class_names=True,
                            data_name='neurons'):
    """ Join spike data from neurons to each trial in maestro_data and convert
    to an output list of NeuronTrial objects. """

    neuron_meta = {}
    default_nums = {}
    default_name = "n_"
    for n_ind, n in enumerate(neurons):
        neuron_meta[n_ind] = {}
        # Need a name for this neuron
        if use_class_names:
            try:
                use_name = neurons[n_ind]['class']
                if use_name is None:
                    # Skip to below
                    raise KeyError()
                else:
                    use_class = use_name
            except KeyError:
                # Neuron does not have a class field so use default
                use_name = default_name
                use_class = None
        else:
            use_name = default_name
        if use_name in default_nums:
            default_nums[use_name] += 1
        else:
            default_nums[use_name] = 0
        neuron_meta[n_ind]['name'] = use_name + "{:02d}".format(default_nums[use_name])
        neuron_meta[n_ind]['class'] = use_class
        neuron_meta[n_ind]['indexer'] = Indexer(n['spike_indices'])

    if dt_data is None:
        # Use sampling rate in ms as dt_data
        dt_data = 1000 / neurons[0]['sampling_rate__']
    samples_per_ms = (neurons[0]['sampling_rate__'] / 1000)
    # Convert spike time cushion from ms to samples
    trial_list = []
    for t in maestro_data:
        # if type(t['events']) != dict:
        #     raise ValueError("Input trial_dict key 'events' for maestro_data must be a dictionary that maps event name to the time of the event.")
        tdict = {}
        tdict['name'] = t['header']['name']
        tdict['events'] = {}
        tdict['meta_data'] = {}
        tdict['data'] = {}
        # Get trial stop in units of samples to match the neuron spikes
        # Base on trial stop since it was more accurate when XS2 was screwey
        stop_sample = int(np.ceil(t['plexon_start_stop'][1] * samples_per_ms)) # convert ms to samples
        # Need to keep the trial duration matched to Maestro duration, so just
        # find trial start assuming Maestro is true since pl2 has higher sampling
        start_sample = stop_sample - (t['header']['_num_saved_scans'] * samples_per_ms) # convert ms to samples
        # Need to make dt_duration match maestro duration in whole ms
        dt_duration = int( ((stop_sample - start_sample) / samples_per_ms) / dt_data)
        if dt_duration * dt_data != t['header']['_num_saved_scans']:
            raise ValueError("Plexon data duration conversion to ms {0} does not match maestro scans {1}.".format(dt_duration * dt_data, t['header']['_num_saved_scans']))
        for n_ind, n in enumerate(neurons):
            # Initate the neuron dictionary for this trial and this neuron
            tdict['meta_data'][neuron_meta[n_ind]['name']] = {}
            tdict['meta_data'][neuron_meta[n_ind]['name']]['class'] = neuron_meta[n_ind]['class']
            if n['spike_indices'][0] > stop_sample:
                # This neuron has no spikes within this trial window
                tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] = None
                continue

            ind_spikes_stop = neuron_meta[n_ind]['indexer'].move_index_next(stop_sample, ">")
            if ind_spikes_stop is None:
                # Should be the end of the spikes so reset index to find previous
                neuron_meta[n_ind]['indexer'].set_current_index(-1)
            ind_spikes_start = neuron_meta[n_ind]['indexer'].move_index_previous(start_sample, "<")

            # Get spikes only within the trial window
            tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] = np.copy(np.float64(n['spike_indices'][ind_spikes_start:ind_spikes_stop]))
            t_spikes_sel = np.logical_and(tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] >= start_sample,
                                          tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] <= stop_sample)
            tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] = tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'][t_spikes_sel]

            if tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'].size == 0:
                # No spikes for this trial
                tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] = None
                continue

            # Adjust indices to trial window starting at 0
            tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] -= start_sample
            # Convert to ms within trial
            tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes'] *= (1000 / n['sampling_rate__'])
            # Need to convert spikes to a timeseries for output in 'data'
            tdict['data'][neuron_meta[n_ind]['name']] = np.zeros(dt_duration, dtype=np.uint16)
            for spk in tdict['meta_data'][neuron_meta[n_ind]['name']]['spikes']:
                spk_bin = int(np.floor(spk / dt_data + 0.5))
                tdict['data'][neuron_meta[n_ind]['name']][spk_bin] += 1

        trial_list.append(NeuronTrial(tdict, dt_data, start_data, data_name))

    return trial_list
