import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line('markers', 'data: needs the dataset (skipped without it)')
    config.addinivalue_line('markers', 'gpu: needs CUDA')


def pytest_collection_modifyitems(config, items):
    import torch
    from sliq.common import paths
    has_data = os.path.exists(os.path.join(paths.MAIN, 'splits.json'))
    for item in items:
        if 'data' in item.keywords and not has_data:
            item.add_marker(pytest.mark.skip(reason='dataset not found'))
        if 'gpu' in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason='no CUDA'))
