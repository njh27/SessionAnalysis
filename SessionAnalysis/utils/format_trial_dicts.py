from SessionAnalysis import trial



def is_compressed(t):
    try:
        _ = t['compressed_target']
        return True
    except KeyError:
        return False


def events_list_to_dict(events_list, event_names=None):
    """ Assigns the list of events of a single Maestro trial to a dictionary.

    If event names are given, only these events are kept in the output. If
    event names is None, all events are named numerically as:
        event_dict[DIO channel][pulse ordered index]

    Parameters
    ----------
    events_list : python list
        The list of events as output in MaestroRead (maestro_data['events']
        with each element being a list corresponding to the pulses of a DIO
        channel. Each internal list value contains the timestamp for each pulse
        on the DIO channel.
    event_names : python dict
        The dictionary contains desired names for events as its keys. Each key
        must return a two element tuple or list of integers that specify the
        DIO channel and ordered event number on that channel corresponding to
        the name of the desired event.

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
    event_dict = {}
    if event_names is None:
        for dio_ind, dio_chan in enumerate(events_list):
            if len(dio_chan) > 0:
                event_dict[dio_ind] = {}
                for evn_ind, evn in enumerate(dio_chan):
                    event_dict[dio_ind][evn_ind] = evn
    else:
        for evn in event_names.keys():
            try:
                event_dict[evn] = maestro_data[511]['events'][event_names[evn][0]][event_names[evn][1]]
            except:
                raise LookupError("Could not find matching event indices for event name '{0}' with indices {1}".format(evn, event_names[evn]))


def maestro_to_trial(maestro_data):
    """ Convert maestro_data to format suitable for making t objects.
    """
    keep_keys = ['horizontal_eye_position',
                 'vertical_eye_position',
                 'horizontal_eye_velocity',
                 'vertical_eye_velocity']

    optn_keys = ['horizontal_target_position',
                 'vertical_target_position',
                 'horizontal_target_velocity',
                 'vertical_target_velocity']
    trial_dicts = []
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
        trial_dicts.append(tdict)

    return trial_dicts


def maestro_to_apparatus_trial(maestro_data, data_name="apparatus"):
    """ Convert maestro_data to format suitable for making ApparatusTrial
    objects. """
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
            tdict['data']['targets'] = t['targets']
        else:
            for key in optn_keys:
                tdict['data'][key] = t[key]



        trial_list.append(trial.ApparatusTrial(tdict, data_name))

    return trial_list


def maestro_to_behavior_trial(maestro_data):
    """ Convert maestro_data to format suitable for making BehaviorTrial
    objects. """
    keep_keys = ['horizontal_eye_position',
                 'vertical_eye_position',
                 'horizontal_eye_velocity',
                 'vertical_eye_velocity']
    trial_dicts = []
    for t in maestro_data:
        tdict = {}
        tdict['name'] = t['header']['name']
        tdict['events'] = t['events']
        tdict['data'] = {}
        for key in keep_keys:
            tdict['data'][key] = t[key]
        trial_dicts.append(tdict)

    return trial_dicts
