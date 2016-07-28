import sys
from tqdm import tqdm, tqdm_notebook


if 'ipykernel' in sys.modules:
    progress_bar = tqdm_notebook
else:
    progress_bar = tqdm