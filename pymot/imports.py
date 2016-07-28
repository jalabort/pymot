import numpy as np
from pymot.utils import progress_bar

# TODO(jalabort): Should we include the ball?
def import_tracab(tracab_data_path, tracab_video_path='', start=0,
                  end=np.iinfo(np.int).max, bb_height=10, bb_width=10):
    r"""
    Import Tracab data file containing ground truth tracking annotations.

    Parameters
    ----------
    tracab_data_path : `str`
        Path to the Tracab data file.
    tracab_video_path : `str`, optional
        Path to the Tracab video that the Tracab data describes.
    start : ```int` >= 0``, optional
        Frame number at which to start importing the tracab data.
    end : ```int` >= 0``, optional
        Frame number at which to stop importing the Tracab data.
    bb_height : ```int` > 0``, optional
        Default bounding box height.
    bb_width : ```int` > 0``, optional
        Default bounding box height.
    Returns
    -------
    annotations : `dict`
        The dictionary representing the ground truth tracking annotations.
    """
    if start < 0:
        raise ValueError('Parameter `start` must be bigger than or equal to 0.')
    if end < start:
        raise ValueError('Parameter `end` must be bigger than `start`.')
    if bb_height < 1 or bb_width < 1:
        raise ValueError('Parameters `bb_height` and `bb_width` must be '
                         'bigger than or equal to 0.')

    with open(tracab_data_path) as tracab_data_file:
        annotations = {'class': 'video',
                       'type': 'panoramic',
                       'file_path': tracab_video_path}

        frames = []
        # For every line in the tracab data file
        for i, line in progress_bar(enumerate(tracab_data_file)):

            if i < start:
                continue

            if end <= i:
                break

            # Split the line in different chunks:
            # chunk 1: frame number
            # chunk 2: player/referee data separated by ';'
            # chunk 3: ball data
            data_chunks = line.split(':')

            # Tracab does not provide timestamps, only frame numbers
            f = {'class': 'frame',
                 'timestamp': -1,
                 'num': data_chunks[0]}

            # Split chunk 2 in different players/referees
            targets = data_chunks[1].split(';')[:-1]

            bounding_boxes = []
            # For each player/referee
            for t in targets:
                # Split its data in its different parts:
                # team, target id, jersey number, position x, position y, speed
                t_data = t.split(',')
                b = {'id': t_data[1],
                     'y': int(t_data[4]),
                     'x': int(t_data[3]),
                     'height': bb_height,
                     'width': bb_width}
                bounding_boxes.append(b)

            f['annotations'] = bounding_boxes
            frames.append(f)

        annotations['frames'] = frames

        return annotations
