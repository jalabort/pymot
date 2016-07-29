import numpy as np
from pymot.utils import progress_bar


def mock_detections(annotations, y_min, y_max, x_min, x_max, noise_factor=0.1,
                    false_positive_rate=0.1, false_negative_rate=0.1):
    r"""
    Generate mock detections from a dictionary representing ground truth
    tracking annotations.

    Parameters
    ----------
    annotations : `dict`
        Dictionary representing ground truth tracking annotations.
        Should have the following format:
    y_min: `int`
        Minimum allowed value along the vertical axis of the coordinate
        system  where the annotations are defined.
    y_max: `int`
        Maximum allowed value along the vertical axis of the coordinate
        system  where the annotations are defined.
    x_min: `int`
        Minimum allowed value along the horizontal axis of the coordinate
        system  where the annotations are defined.
    x_max: `int`
        Maximum allowed value along the horizontal axis of the coordinate
        system  where the annotations are defined.
    noise_factor: `float`, optional
        Controls the amount of noise used to simulate the bounding boxes
        produced by the detector. The higher this value the more noisy the
        simulated bounding box detections will be. If `0` the simulated
        detection bounding boxes will match the ones on the annotations.
    false_positive_rate : `float`, optional
        Determines the probability of simulating a false positive, i.e the
        probability of generating a simulated bounding box where there
        should not be one.
    false_negative_rate :  `float`, optional
        Determines the probability of simulating a false negative, i.e the
        probability of not generating a simulated bounding box where there
        should be one.

    Returns
    -------
    detections: `dict`
        The dictionary representing simulated detection.
    """
    detections = {'class': annotations['class'],
                  'type': annotations['type'],
                  'file_path': annotations['type']}

    frames = []
    for a_frame in progress_bar(annotations['frames']):

        f = {'class': 'frame',
             'timestamp': a_frame['timestamp'],
             'num': a_frame['num']}

        bounding_boxes = []
        for a_b in a_frame['annotations']:

            if np.random.uniform() > false_negative_rate:
                y_noise = noise_factor * a_b['height']
                x_noise = noise_factor * a_b['width']
                y = a_b['y'] + y_noise * np.random.randn(1)
                x = a_b['x'] + x_noise * np.random.randn(1)
                height = a_b['height'] + y_noise * np.random.randn(1)
                width = a_b['width'] + x_noise * np.random.randn(1)

                b = {'id': -1,
                     'y': int(y),
                     'x': int(x),
                     'height': np.maximum(int(height), 1),
                     'width': np.maximum(int(width), 1)}

                bounding_boxes.append(b)

            if np.random.uniform() < false_positive_rate:
                y = np.random.uniform(y_min, y_max)
                x = np.random.uniform(x_min, x_max)
                y_noise = noise_factor * a_b['height']
                x_noise = noise_factor * a_b['width']
                height = a_b['height'] + y_noise * np.random.randn(1)
                width = a_b['width'] + x_noise * np.random.randn(1)

                b = {'id': -1,
                     'y': int(y),
                     'x': int(x),
                     'height': np.maximum(int(height), 1),
                     'width': np.maximum(int(width), 1)}

                bounding_boxes.append(b)

        f['detections'] = bounding_boxes
        frames.append(f)

    detections['frames'] = frames

    return detections


def _mock_hypotheses(annotations, y_min, y_max, x_min, x_max, noise_factor=0.1,
                     false_positive_rate=0.1, false_negative_rate=0.1,
                     identity_switch_rate=0.1):
    r"""
    Generate mock hypotheses from a dictionary representing ground truth
    tracking annotations. This is mainly useful for manual testing of the
    map:`MotEvaluation` class.

    Parameters
    ----------
    annotations : `dict`
        Dictionary representing ground truth tracking annotations.
        Should have the following format:
    y_min: `int`
        Minimum allowed value along the vertical axis of the coordinate
        system  where the annotations are defined.
    y_max: `int`
        Maximum allowed value along the vertical axis of the coordinate
        system  where the annotations are defined.
    x_min: `int`
        Minimum allowed value along the horizontal axis of the coordinate
        system  where the annotations are defined.
    x_max: `int`
        Maximum allowed value along the horizontal axis of the coordinate
        system  where the annotations are defined.
    noise_factor: `float`, optional
        Controls the amount of noise used to simulate the bounding boxes
        produced by the detector. The higher this value the more noisy the
        simulated bounding box detections will be. If `0` the simulated
        detection bounding boxes will match the ones on the annotations.
    false_positive_rate : `float`, optional
        Determines the probability of simulating a false positive, i.e the
        probability of generating a simulated bounding box where there
        should not be one.
    false_negative_rate : `float`, optional
        Determines the probability of simulating a false negative, i.e the
        probability of not generating a simulated bounding box where there
        should be one.
    identity_switch_rate : `float`, optional
        Determines the probability of simulating an identity switch.

    Returns
    -------
    detections: `dict`
        The dictionary representing simulated detection.
    """
    hypotheses = {'class': annotations['class'],
                  'type': annotations['type'],
                  'file_path': annotations['type']}

    frames = []
    for a_frame in progress_bar(annotations['frames']):

        f = {'class': 'frame',
             'timestamp': a_frame['timestamp'],
             'num': a_frame['num']}

        last_index = len(a_frame['annotations']) - 1
        idsw = False
        bounding_boxes = []
        for i, a_b in enumerate(a_frame['annotations']):

            if idsw:
                y_noise = noise_factor * a_b['height']
                x_noise = noise_factor * a_b['width']
                y = a_b['y'] + y_noise * np.random.randn(1)
                x = a_b['x'] + x_noise * np.random.randn(1)
                height = a_b['height'] + y_noise * np.random.randn(1)
                width = a_b['width'] + x_noise * np.random.randn(1)

                b = {'id': idsw,
                     'y': int(y),
                     'x': int(x),
                     'height': np.maximum(int(height), 1),
                     'width': np.maximum(int(width), 1)}

                bounding_boxes.append(b)

                idsw = False
                continue

            if np.random.uniform() < false_positive_rate:
                y = np.random.uniform(y_min, y_max)
                x = np.random.uniform(x_min, x_max)
                y_noise = noise_factor * a_b['height']
                x_noise = noise_factor * a_b['width']
                height = a_b['height'] + y_noise * np.random.randn(1)
                width = a_b['width'] + x_noise * np.random.randn(1)

                b = {'id': a_b['id'],
                     'y': int(y),
                     'x': int(x),
                     'height': np.maximum(int(height), 1),
                     'width': np.maximum(int(width), 1)}

                bounding_boxes.append(b)
                continue

            if np.random.uniform() < identity_switch_rate and i < last_index:
                y_noise = noise_factor * a_b['height']
                x_noise = noise_factor * a_b['width']
                y = a_b['y'] + y_noise * np.random.randn(1)
                x = a_b['x'] + x_noise * np.random.randn(1)
                height = a_b['height'] + y_noise * np.random.randn(1)
                width = a_b['width'] + x_noise * np.random.randn(1)

                b = {'id': a_frame['annotations'][i + 1]['id'],
                     'y': int(y),
                     'x': int(x),
                     'height': np.maximum(int(height), 1),
                     'width': np.maximum(int(width), 1)}

                bounding_boxes.append(b)

                idsw = a_b['id']
                continue

            if np.random.uniform() > false_negative_rate:
                y_noise = noise_factor * a_b['height']
                x_noise = noise_factor * a_b['width']
                y = a_b['y'] + y_noise * np.random.randn(1)
                x = a_b['x'] + x_noise * np.random.randn(1)
                height = a_b['height'] + y_noise * np.random.randn(1)
                width = a_b['width'] + x_noise * np.random.randn(1)

                b = {'id': a_b['id'],
                     'y': int(y),
                     'x': int(x),
                     'height': np.maximum(int(height), 1),
                     'width': np.maximum(int(width), 1)}
                
                bounding_boxes.append(b)

        f['hypotheses'] = bounding_boxes
        frames.append(f)

    hypotheses['frames'] = frames

    return hypotheses