import warnings


def check_format(groundtruth, hypotheses):
    r"""
    Function used to check the format of groundtruth and hypotheses
    dictionaries.

    :param groundtruth: `dict`
        Dictionary representing groundtruth tracking annotations.
        Should have the following format:
    :param hypotheses: `dict`
        Dictionary representing tracking hypotheses.
        Should have the following format:

    :return: `(bool, bool)` `tuple` of `bool`
        The first element of the tuple is `True` if the ground truth format is
        correct, `False` otherwise. The second element of the is `True` if
        the hypotheses format is correct, `False` otherwise.
    """
    return _check(groundtruth, 'groundtruth'), _check(hypotheses, 'hypotheses')


def _check(dic, mode):
    r"""
    Private function used to check the format of groundtruth and hypotheses
    dictionaries.

    :param dic: `dict`
        Dictionary representing either ground truth tracking annotations or
        tracking hypotheses. Should have the following format:
    :param mode: `str`
        `groundtruth` if the previous dictionary represents ground truth
        tracking annotations. `hypotheses` if the previous dictionary
        represents tracking hypotheses.

    :return: `bool`
        `True` if the dictionary's format is correct.
    """
    if mode == 'groundtruth':
        type_key = 'annotations'
        type_name = 'Ground truth'
    elif mode == 'hypotheses':
        type_key = 'hypotheses'
        type_name = 'Hypotheses'
    else:
        raise ValueError('Argument mode must be 0 (for ground truth) or 1 '
                         '(for hypotheses)')

    result = True

    for frame in dic['frames']:
        ids = set()
        for bb in frame[type_key]:
            if not 'id' in bb:
                warnings.warn(
                    '%s without id found, timestamp %f, frame %d!'
                    % (type_name, frame['timestamp'],
                       frame['num'] if 'num' in frame else -1))
                result &= False
            elif bb['id'] == '':
                warnings.warn(
                    '%s with empty id found, timestamp %f, frame %d!'
                    % (type_name, frame['timestamp'],
                       frame['num'] if "num" in frame else -1))
                result &= False
            elif bb['id'] in ids:
                warnings.warn(
                    '%s with ambiguous id (%s) found, timestamp %f, frame %d'
                    % (type_name, str(bb['id']), frame['timestamp'],
                       frame['num'] if 'num' in frame else -1))
                result &= False
            else:
                ids.add(bb['id'])

            for key in ['x', 'y', 'width', 'height']:
                if not key in bb.keys():
                    warnings.warn(
                        '%s without key %s found, timestamp %f, frame %d!'
                        % (type_name, key, frame['timestamp'],
                           frame['num'] if "num" in frame else -1))
                    result &= False

    return result
