import collections

import numpy as np
import pytest
import torch
from PIL import Image

from sliq.classifier import dataset as D


pytestmark = pytest.mark.data


@pytest.fixture(scope='module')
def train_samples():
    return D.load_samples('train')


def test_sample_counts(train_samples):
    counts = collections.Counter((D.CLASSES[s['cls']], s['origin']) for s in train_samples)
    assert counts == {('On', 'real'): 1824, ('Dim', 'real'): 75, ('Dim', 'synthetic'): 250, ('Off', 'synthetic'): 400}
    assert [len(D.load_samples(split)) for split in ('val', 'test')] == [430, 236]


def test_crop_shape(train_samples):
    x, _, _ = D.ClassifierDataset(train_samples, augment=True)[0]
    assert tuple(x.shape) == (3, 224, 224)
    x, _, synthetic = D.ClassifierDataset(train_samples[-3:])[0]
    assert synthetic == 1 and x.dtype == torch.float32


def test_unaugmented_crop_matches_the_exported_crop():
    samples = [s for s in D.load_samples('val') if s['origin'] == 'real'][:5]
    dataset = D.ClassifierDataset(samples)
    for i, sample in enumerate(samples):
        first, second = dataset[i][0], dataset[i][0]
        assert torch.equal(first, second)
        exported = Image.open(f"{D.DATASET}/classifier/val/{D.CLASSES[sample['cls']]}/{sample['name']}.png").convert('RGB')
        assert (first - D.to_tensor(exported)).abs().mean() < 0.15          # box edge rounding


def test_augmentation_moves_geometry_not_brightness(train_samples):
    real = [s for s in train_samples if s['origin'] == 'real'][:30]
    plain, augmented = D.ClassifierDataset(real), D.ClassifierDataset(real, augment=True, seed=1)
    differences = [(augmented[i][0] - plain[i][0]).abs().mean().item() for i in range(30)]
    assert max(differences) > 0.01 and min(differences) == pytest.approx(0, abs=1e-6)
    assert abs(np.mean([plain[i][0].mean().item() for i in range(30)]) - np.mean([augmented[i][0].mean().item() for i in range(30)])) < 0.1


def test_sampler_composition(train_samples):
    drawn = np.random.RandomState(0).choice(len(train_samples), 20000, p=D.sampler_weights(train_samples))
    share = collections.Counter((D.CLASSES[train_samples[i]['cls']], train_samples[i]['origin']) for i in drawn)
    assert abs(share[('On', 'real')] / 20000 - 1 / 3) < 0.02 and abs(share[('Off', 'synthetic')] / 20000 - 1 / 3) < 0.02
    assert abs(share[('Dim', 'real')] / 20000 - 1 / 6) < 0.02 and abs(share[('Dim', 'synthetic')] / 20000 - 1 / 6) < 0.02
