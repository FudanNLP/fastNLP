
import pytest

from fastNLP import DataSet, TorchDataLoader
from fastNLP.core.dataloaders.torch_dataloader.fdl import (
    prepare_torch_dataloader,
)


@pytest.fixture
def dataset():
    return DataSet({
        "x": [1, 2, 3, 4],
        "y": [0, 1, 0, 1],
    })


def test_default_num_workers_zero(dataset):
    """Default settings should work without multiprocessing."""
    loader = TorchDataLoader(dataset, batch_size=2)

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert len(list(loader)) == 2


def test_explicit_none(dataset):
    """Explicit None should be accepted with zero workers."""
    loader = TorchDataLoader(
        dataset,
        num_workers=0,
        prefetch_factor=None,
    )

    assert loader.prefetch_factor is None


def test_invalid_prefetch_factor(dataset):
    """Explicit non-None prefetching with zero workers is invalid."""
    with pytest.raises(ValueError, match="prefetch_factor"):
        TorchDataLoader(
            dataset,
            num_workers=0,
            prefetch_factor=2,
        )


def test_prepare_dataloader_default(dataset):
    """The convenience function should use valid defaults."""
    loader = prepare_torch_dataloader(dataset, batch_size=2)

    assert loader.prefetch_factor is None
    assert len(list(loader)) == 2


def test_multiprocessing_prefetch_default(dataset):
    """Multiple workers should retain the default prefetch factor."""
    loader = TorchDataLoader(
        dataset,
        num_workers=2,
        multiprocessing_context="spawn",
    )

    assert loader.num_workers == 2
    assert loader.prefetch_factor == 2


def test_prepare_multiple_datasets(dataset):
    loaders = prepare_torch_dataloader({
        "train": dataset,
        "dev": dataset,
    })

    assert loaders["train"].prefetch_factor is None
    assert loaders["dev"].prefetch_factor is None
