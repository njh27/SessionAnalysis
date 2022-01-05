

def is_compressed(trial):
    try:
        _ = trial['compressed_target']
        return True
    except KeyError:
        return False


def maestro_to_trial_dict(maestro_data):
    """ Convert maestro_data to format suitable for making trial objects.
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
    for trial in maestro_data:
        tdict = {}
        tdict['name'] = trial['header']['name']
        tdict['events'] = trial['events']
        tdict['data'] = {}
        if is_compressed(trial):
            for key in keep_keys:
                tdict['data'][key] = trial[key]
            tdict['data']['targets'] = trial['targets']
        else:
            for key in keep_keys:
                tdict['data'][key] = trial[key]
            for key in optn_keys:
                tdict['data'][key] = trial[key]
        trial_dicts.append(tdict)

    return trial_dicts


def maestro_to_apparatus_trial_dict(maestro_data):
    """ Convert maestro_data to format suitable for making ApparatusTrial
    objects. """
    optn_keys = ['horizontal_target_position',
                 'vertical_target_position',
                 'horizontal_target_velocity',
                 'vertical_target_velocity']
    trial_dicts = []
    for trial in maestro_data:
        tdict = {}
        tdict['name'] = trial['header']['name']
        tdict['events'] = trial['events']
        tdict['data'] = {}
        if is_compressed(trial):
            tdict['data']['targets'] = trial['targets']
        else:
            for key in optn_keys:
                tdict['data'][key] = trial[key]
        trial_dicts.append(tdict)

    return trial_dicts


def maestro_to_behavior_trial_dict(maestro_data):
    """ Convert maestro_data to format suitable for making BehaviorTrial
    objects. """
    keep_keys = ['horizontal_eye_position',
                 'vertical_eye_position',
                 'horizontal_eye_velocity',
                 'vertical_eye_velocity']
    trial_dicts = []
    for trial in maestro_data:
        tdict = {}
        tdict['name'] = trial['header']['name']
        tdict['events'] = trial['events']
        tdict['data'] = {}
        for key in keep_keys:
            tdict['data'][key] = trial[key]
        trial_dicts.append(tdict)

    return trial_dicts
