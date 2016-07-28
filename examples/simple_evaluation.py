import json
from pymot import MotEvaluation, check_format


annotations_file = '../data/annotations.json'
hypotheses_file = '../data/hypotheses.json'



a = open(annotations_file)

if annotations_file.endswith('.json'):
    annotations = json.load(a)
else:
    raise Exception('Only json is supported!')

a.close()



h = open(hypotheses_file)

if hypotheses_file.endswith('.json'):
    hypotheses = json.load(h)
else:
    raise Exception('Only json is supported!')

h.close()



a_correct, h_correct = check_format(annotations, hypotheses)
if not a_correct :
    raise ValueError('Format of file %s is incorrect!') % annotations_file
elif not h_correct :
    raise ValueError('Format of file %s is incorrect!') % hypotheses_file



evaluator = MotEvaluation(annotations, hypotheses)
evaluator.evaluate()


print('Track statistics')
evaluator.print_track_statistics()

print('')

print('Results')
evaluator.print_results()