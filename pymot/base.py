import warnings
import json
import numpy as np
from munkres import Munkres
from pymot.boundingbox import BoundingBox


# TODO(jalabort): Can we have this class work for both intersection over union and center distance effectively?
class MotEvaluation:
    r"""
    Class used to evaluate the performance of Multi-Object Tracking (MOT)
    algorithms.

    Parameters
    ----------
    annotations : `dict`
        Dictionary representing ground truth tracking annotations.
        Should have the following format:
    hypotheses : `dict`
        Dictionary representing tracking hypotheses.
        Should have the following format:
    overlap_threshold : `float`, optional
        Bounding box overlap threshold. Threshold below which a hypothesis
        previously associated to a ground truth target is considered to have
        drift and started tracking something else.
    sync_delta: `float`, , optional
        Maximum time offset considered for matching hypotheses and ground
        truth annotations.
    munkres_inf: `float`, optional
        Not quite infinite number used by the Hungarian algorithm by
        default. Should be set to a large floating point number

    References
    ----------
    .. [1] A. Milan, et al., "MOT16: A Benchmark for Multi-Object Tracking",
        arXiv 2016.
        http://arxiv.org/abs/1603.00831
    .. [2] K. Bernardin and R. Stiefelhagen, "Evaluating Multiple Object
        Tracking Performance: The CLEAR MOT Metrics", TPAMI 2008.
        http://jivp.eurasipjournals.springeropen.com/articles/10.1155/2008/246309
    """
    def __init__(self, annotations, hypotheses, overlap_threshold = 0.2,
                 sync_delta = 0.001, munkres_inf=np.finfo(np.float).max):
        self._annotations = annotations
        self._hypotheses = hypotheses

        self._overlap_threshold = overlap_threshold
        self._sync_delta = sync_delta
        self._munkres_inf = munkres_inf

        # Set initial state
        self._reset_mapping()
        self._reset_statistics()
        self._reset_annotations()
        self._visual_debug_frames = []
        self._evaluated = False


    def reset(self):
        r"""
        Reset the state of the `:map:MotEvaluation` object. Current evaluation
        data will be deleted and the object's state will match its initial
        state after construction.
        """
        self._reset_mapping()
        self._reset_statistics()
        self._reset_annotations()
        self._visual_debug_frames = []
        self._evaluated = False

    def _reset_mapping(self):
        self._mappings = {}

        self._a_map = {}
        self._h_map = {}

    def _reset_statistics(self):
        self._false_negatives = 0
        self._false_positives = 0
        self._identity_switches = 0
        self._total_annotations = 0

        self._total_overlap = 0.0
        self._total_correspondences = 0

        self._annotations_ids = set()
        self._hypotheses_ids = set()

    def _reset_annotations(self):
        for f in self._annotations['frames']:
            for a in f['annotations']:
                a['type'] = 'annotations'
                a['class'] = 'unevaluated'

        for f in self._hypotheses['frames']:
            for h in f['hypotheses']:
                h['type'] = 'hypothesis'
                h['class'] = 'unevaluated'

    @property
    def annotations(self):
        r"""
        The ground truth tracking annotations.

        :type: `dict`
        """
        return self._annotations

    @property
    def hypotheses(self):
        r"""
        The tracking hypotheses.

        :type: `dict`
        """
        return self._hypotheses

    @property
    def overlap_threshold(self):
        r"""
        The necessary bounding box overlap between hypotheses and
        annotations for them to be considered correspondences.

        :type: `float`
        """
        return self._overlap_threshold

    @property
    def sync_delta(self):
        r"""
        The temporal resolution threshold below which hypotheses and
        annotation frames are considered to be chronologically close and
        can be safely matched.

        :type: `float`
        """
        return self._sync_delta

    @property
    def evaluated(self):
        r"""
        Flag indicating whether the method `self.evaluate()` has been called
        or not and, hence, indicating if the evaluation has been carried out.

        :type: `bool`
        """
        return self._evaluated

    @property
    def total_annotations(self):
        r"""
        The total number of ground truth annotated targets.

        :type: `int`
        """
        return self._total_annotations

    @property
    def false_negatives(self):
        r"""
        The total number of false negatives (misses).

        :type: `int`
        """
        return self._false_negatives

    @property
    def false_positives(self):
        r"""
        The total number of false positives.

        :type: `int`
        """
        return self._false_positives

    @property
    def identity_switches(self):
        r"""
        The total number of identity switches (mismatches).

        :type: `int`
        """
        return self._identity_switches

    @property
    def correspondences(self):
        r"""
        The total number of valid correspondences between ground
        truth tracking annotations and tracking hypotheses.

        :type: `int`
        """
        return self._total_correspondences

    @property
    def overlap(self):
        r"""
        The total sum of the IoU between the bounding boxes of all valid
        annotation-hypothesis correspondences.

        :type: `float`
        """
        return self._total_overlap

    @property
    def false_negative_rate(self):
        r"""
        The false negative (misses) rate.

        :type: `int`
        """
        return float(self._false_negatives) / self._total_annotations

    @property
    def false_positive_rate(self):
        r"""
        The false positive rate.

        :type: `int`
        """
        return float(self._false_positives) / self._total_annotations

    @property
    def identity_switch_rate(self):
        r"""
        The identity switch (mismatch) rate.

        :type: `int`
        """
        return float(self._identity_switches) / self._total_annotations

    @property
    def tracking_precision(self):
        r"""
        """
        return (float(len(self._h_map.keys())) / len(self._hypotheses_ids)
                if len(self._hypotheses_ids) != 0 else 0.0)

    @property
    def tracking_recall(self):
        r"""
        """
        return (float(len(self._a_map.keys())) / len(self._annotations_ids)
                if len(self._annotations_ids) != 0 else 0.0)

    @property
    def lonely_hypotheses_tracks(self):
        r"""
        """
        return len(self._hypotheses_ids - set(self._h_map.keys()))

    @property
    def hypotheses_tracks(self):
        r"""
        """
        return len(self._hypotheses_ids)

    @property
    def lonely_annotation_tracks(self):
        r"""
        """
        return len(self._annotations_ids - set(self._a_map.keys()))

    @property
    def annotation_tracks(self):
        r"""
        """
        return len(self._annotations_ids)

    @property
    def covered_annotation_tracks(self):
        r"""
        """
        return self._annotations_ids & set(self._a_map.keys())

    @property
    def covered_hypothesis_tracks(self):
        r"""
        """
        return len(self._h_map.keys())

    @property
    def visual_debug(self):
        r"""
        """
        return {'filename': self._annotations['filename'],
                'class': self._annotations['class'],
                'frames': self._visual_debug_frames}

    def mota(self):
        r"""
        Get the Multi Object Tracking Accuracy (MOTP) metric.

        Returns
        -------
        mota : `float`
            Value between ``(-Inf, 1.0]`` representing the MOTA metric.
        """
        return calculate_mota(self._false_negatives, self._false_positives,
                              self._identity_switches, self._total_annotations)

    def motp(self):
        r"""
        Get the Multi Object Tracking Precision (MOTP) metric.

        Returns
        -------
        motp : `float`
            Value between ``[0, 1.0]`` representing the MOTP metric.
        """
        return calculate_motp(self._total_overlap, self._total_correspondences)

    def evaluate(self):
        r"""
        Compute Multi-Object Tracking (MOT) performance metrics.
        """
        frames = self._annotations['frames']
        for frame in frames:
            self._evaluate_frame(frame)
        self._evaluated = True

    def _evaluate_frame(self, frame):
        r"""
        Update performance metrics by evaluating a new frame.

        Parameters
        ----------
        frame: `dict`
            Dictionary representing the ground truth tracking annotations
            of the new frame.
        """
        timestamp = frame['timestamp']
        annotations = frame['annotations']
        hypotheses = self._hypotheses_frame(timestamp)['hypotheses']

        visual_debug_annotations = []

        for a in annotations:
            self._annotations_ids.add(a['id'])

        for h in hypotheses:
            self._hypotheses_ids.add(h['id'])

        if len(annotations) == 0 and len(hypotheses) == 0:
            return

        # Step 1
        # Check if previous correspondences still hold on the current frame.
        correspondences = {}

        for a_id in self._mappings.keys():
            a = [a for a in annotations if a['id'] == a_id]

            if len(a) > 1:
                warnings.warn('Found %d > 1 ground truth tracks for id %s',
                              len(a), a_id)
            elif len(a) < 1:
                continue

            h_id = self._mappings[a_id]
            h = [h for h in hypotheses if h['id'] == h_id]

            assert len(h) <= 1
            if len(h) != 1:
                continue

            # Hypothesis found for known mapping, check hypothesis for overlap
            a_bb = BoundingBox.init_from_dic(a[0])
            h_bb = BoundingBox.init_from_dic(h[0])

            # TODO(jalabort): Should overlap to be computed using a function handle provided at construction time?
            overlap = a_bb.intersection_over_union(h_bb)

            if overlap >= self._overlap_threshold:
                correspondences[a_id] = h[0]['id']
                self._total_overlap += overlap


        # Step 2
        # Find remaining correspondences using the Hungarian algorithm

        # Build distance matrix and fill all of its entries with a not quite
        # infinite number
        distance_matrix = np.empty((len(hypotheses), len(annotations)))
        distance_matrix.fill(self._munkres_inf)

        # Fill out the previous distance matrix with true overlaps between
        # annotations and hypothesis. Note that possible valid correspondences
        # found in step one will be ignored (this is because temporal
        # coherence precedes bounding box overlap in this case).
        # For all annotations
        for i, a in enumerate(annotations):
            # If the annotations has not being assigned to a hypothesis yet
            if a['id'] not in correspondences.keys():
                # Iterate over all hypotheses
                for j, h in enumerate(hypotheses):
                    # If the hypothesis has not being assigned to an
                    # annotation yet.
                    if h['id'] not in correspondences.values():
                        # Compute their overlap
                        a_bb = BoundingBox.init_from_dic(a)
                        h_bb = BoundingBox.init_from_dic(h)
                        overlap = a_bb.intersection_over_union(h_bb)
                        # If overlap is bigger or equal than the threshold
                        if overlap >= self._overlap_threshold:
                            # Assign the overlap inverse as the distance
                            # between this annotation-hypothesis pair
                            distance_matrix[i][j] = 1.0 / overlap

        # Make sure distance matrix is filled out
        if distance_matrix.shape[0] > 0 and distance_matrix.shape[1] > 0:
            # Run Hungarian algorithm. Returns a list of tuples containing
            # optimally (based on distance) paired annotations and hypothesis
            indices = Munkres().compute(distance_matrix.tolist())
        else:
            indices = []

        # For every pair of optimally associated annotations and hypothesis
        for (a_index, h_index) in indices:
            distance = distance_matrix[a_index][h_index]

            if distance < self._munkres_inf:
                a_id = annotations[a_index]['id']
                h_id = hypotheses[h_index]['id']

                # Check if the annotation's id was already associated with a
                # previous hypothesis
                if a_id in self._mappings:
                    # If it was, then its id must be different from the id
                    # of the current hypothesis since that should have been
                    # handled in Step 1
                    assert self._mappings[a_id] != h_id

                # Save this new annotation-hypothesis correspondence
                correspondences[a_id] = h_id
                # Add its overlap
                self._total_overlap += 1.0 / distance

                # Update independent annotations and hypotheses mapping
                # dictionaries
                self._a_map[a_id] = h_id
                self._h_map[h_id] = a_id

                # Correspondence contradicts previous mapping. Mark and count as mismatch, if ground truth is not a DCO
                # Iterate over all gt-hypo pairs of mapping, since we have to perform a two way check:
                # Correspondence: A-1
                # Mapping: A-2, B-1
                # We have to detect both forms of conflicts

                # TODO(jalabort): Couldn't this be made more efficient by simply querying the dictionary with a_id and h_id?
                # Iterate over all current annotation-hypothesis mappings
                for mapping_a_id, mapping_h_id in list(self._mappings.items()):

                    # If the previous optimal annotation-hypothesis pair
                    # given by the Hungarian algorithm contradicts a current
                    # annotation-hypothesis mapping
                    if ((mapping_a_id == a_id and mapping_h_id != h_id) or
                            (mapping_a_id != a_id and mapping_h_id == h_id)):

                        # Increment the identity switch counter
                        self._identity_switches += 1

                        # Get annotation and hypothesis with given ids
                        a = annotations[a_index]
                        h = hypotheses[h_index]

                        # Mark them as mismatches
                        a['class'] = 'identity_switch'
                        h['class'] = 'identity_switch'

                        # TODO(jalabort): Is this useful?
                        # Add them to the visual debug annotation list
                        visual_debug_annotations.append(a)
                        visual_debug_annotations.append(h)

                        # Delete current annotation-hypothesis mappings
                        del self._mappings[mapping_a_id]

                # Save (overwrite) new annotation-hypothesis mappings
                self._mappings[a_id] = h_id


        # Step 3
        # Count false negatives and false positive

        # For all annotations
        for a in annotations:
            # If the annotation was not mark as identity switch and its id is
            # in correspondence
            if (a['class'] != 'identity_switch' and
                        a['id'] in correspondences.keys()):
                # Mark it as correspondence
                a['class'] = 'correspondence'
                # Add the annotation to the visual debug annotation list
                visual_debug_annotations.append(a)

            # If the annotation id is not in correspondences
            if a['id'] not in correspondences.keys():
                # Increment the false negative counter and mark it as a
                # false negative
                a['class'] = 'false_negative'
                self._false_negatives += 1
                visual_debug_annotations.append(a)

        # For all hypotheses
        for h in hypotheses:
            # If the hypothesis was not mark as identity switch and its id is
            # in correspondence
            if (h['class'] != 'identity_switch' and
                        h['id'] in correspondences.values()):
                # Mark it as correspondence
                h['class'] = 'correspondence'
                visual_debug_annotations.append(h)

            # If the hypothesis id is not in correspondences
            if h['id'] not in correspondences.values():
                # Increment the false positive counter and mark it as a
                # false negative
                self._false_positives += 1
                h['class'] = "false positive"
                visual_debug_annotations.append(h)


        self._total_correspondences += len(correspondences)
        self._total_annotations += len(annotations)

        visual_debug_frame = {'timestamp': timestamp,
                              'class': frame['class'],
                              'annotations': visual_debug_annotations}
        if 'num' in frame:
            visual_debug_frame['num'] = frame['num']

        self._visual_debug_frames.append(visual_debug_frame)

    def _hypotheses_frame(self, timestamp):
        r"""
        Obtain a hypothesis occurring chronologically close to the given
        timestamp with, at most, `self.sync_delta` time difference.

        Parameters
        ----------
        timestamp: `int` or `float`
            The timestamp to be matched.

        Returns
        -------
        hypotheses: `dict`
            A matching hypothesis for the given timestamp.
        """
        # TODO(jalabort): Is this slow?
        hypotheses_frames = [h for h in self._hypotheses['frames'] if
                             abs(h['timestamp'] - timestamp) < self._sync_delta]

        if len(hypotheses_frames) > 1:
            raise Exception(
                "> 1 hypotheses timestamps found for timestamp %f with sync "
                "delta %f" % (timestamp, self._sync_delta))

        if len(hypotheses_frames) == 0:
            warnings.warn(
                'No hypothesis timestamp found for timestamp %f with sync '
                'delta %f' % (timestamp, self._sync_delta))
            return {"hypotheses": []}

        return hypotheses_frames[0]

    def absolute_statistics(self):
        r"""
        """
        return {"total annotations": self.total_annotations,
                "fn": self.false_negatives,
                "fp": self.false_positives,
                "idsw": self.identity_switches,
                "correspondences": self.correspondences,
                "overlap": self.overlap,
                "lonely annotation tracks": self.lonely_hypotheses_tracks,
                "covered annotation tracks": self.covered_annotation_tracks,
                "annotation tracks": self.annotation_tracks,
                "lonely hypothesis tracks": self.lonely_hypotheses_tracks,
                "covered hypothesis tracks": self.covered_hypothesis_tracks,
                "hypothesis tracks": self.hypotheses_tracks}

    def relative_statistics(self):
        return {'fn rate': self.false_negative_rate,
                'fp rate': self.false_positive_rate,
                'idsw rate' : self.identity_switch_rate,
                'mota': self.mota(),
                'motp': self.motp(),
                'tracking precision': self.tracking_precision,
                'tracking recall': self.tracking_recall}

    def print_track_statistics(self):
        r"""
        """
        print('Lonely annotation tracks:  %d' % self.lonely_annotation_tracks)
        print('Annotation tracks:         %d' % self.annotation_tracks)
        print('Lonely hypothesis tracks:  %d' % self.lonely_hypotheses_tracks)
        print('Hypothesis tracks:         %d' % self.hypotheses_tracks)

    def print_results(self):
        r"""
        """
        print('Ground truths:             %d' % self._total_annotations)
        print('False positives:           %d' % self._false_positives)
        print('Misses:                    %d' % self._false_negatives)
        print('Mismatches:                %d' % self._identity_switches)
        print('Correspondences:           %d' % self._total_correspondences)
        print('MOTA:                      %.2f' % self.mota())
        print('MOTP:                      %.2f' % self.motp())

    def print_annotations(self):
        print(json.dumps(self.annotations, sort_keys=True, indent=2))

    def print_hypotheses(self):
        print(json.dumps(self.hypotheses, sort_keys=True, indent=2))

    def print_visual_debug(self):
        print(json.dumps(self.visual_debug, sort_keys=True, indent=2))


def calculate_mota(false_negatives, false_positives, identity_switches,
                   groundtruths):
    r"""
    Compute the Multi Object Tracking Accuracy (MOTA) metric.

    Parameters
    ----------
    false_negatives : `int`
        Total number of false negative (missed) tracking hypothesis.
    false_positives : `int`
        Total number of false positive tracking hypothesis.
    identity_switches : `int`
        Total number of tracking hypothesis that lead to identity switches.
    groundtruths : `int`
        Total number of ground truth targets that should have been tracked.

    Returns
    -------
    mota : `float`
        Value between ``(-Inf, 1.0]`` representing the MOTA metric.
    """
    if groundtruths <= 0:
        raise ValueError(
            'The total number of ground truth targets should be bigger than '
            '0. MOTA calculation not possible.')

    if false_negatives < 0 or false_positives < 0 or identity_switches < 0:
        raise ValueError(
            'The number of false negative (%d), false positives (%d) and '
            'identity switches (%d) must all be bigger or equal than 0.')

    return 1.0 - float(
        false_negatives + false_positives + identity_switches) / groundtruths


def calculate_motp(overlap, correspondences):
    r"""
    Compute the Multi Object Tracking Precision (MOTP) metric.

    Parameters
    ----------
    overlap : `float`
        Total number of tracking hypothesis that lead to identity switches.
    correspondences : `int`
        Total number of ground truth targets that should have been tracked.

    Returns
    -------
    motp : `float`
        Value between ``[0, 1.0]`` representing the MOTP metric.
    """
    if correspondences <= 0:
        raise ValueError(
            'The total number of correspondences should be bigger than '
            '0. MOTP calculation not possible.')

    if overlap < 0:
        raise ValueError(
            'The total overlap (%f) must be bigger or equal than 0.')

    return float(overlap) / correspondences
