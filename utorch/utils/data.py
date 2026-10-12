from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from utorch.autograd import Tensor


class Dataset:
    """An abstract class representing a dataset.

    All custom datasets must override `__getitem__` (returning a sample at an
    index) and `__len__` (returning the total size of the dataset).
    """

    def __getitem__(self, index: int) -> Any:
        """Override this method to return a sample at an index."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Override this method to return the total size of the dataset."""
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors where indexing retrieves slices along axis 0.

    Args:
        *tensors: Tensors or NumPy arrays with identical length along axis 0.
    """

    def __init__(self, *tensors: Any) -> None:
        """Initialize a TensorDataset."""
        assert len(tensors) > 0, "TensorDataset requires at least one tensor."
        first_len = len(tensors[0])
        for t in tensors:
            assert len(t) == first_len, (
                "All tensors must have identical size along dim 0."
            )

        self.tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """Return a single tensor example from the dataset."""
        return tuple(Tensor(t.value[index]) for t in self.tensors)

    def __len__(self) -> int:
        """Return the number of tensors in the dataset."""
        return len(self.tensors[0].value)


class DataLoader:
    """Data loader combining a dataset and sampler, yielding batched Tensors.

    Args:
        dataset: Dataset from which to load data.
        batch_size: How many samples per batch to load (default: 1).
        shuffle: If True, reshuffles indices at the start of every epoch.
        drop_last: If True, drops the last incomplete batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        """Initialize a DataLoader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Return an iterator over the samples in the dataset."""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            samples = [self.dataset[i] for i in batch_indices]
            num_fields = len(samples[0])
            batch = tuple(
                Tensor(np.stack([s[f].value for s in samples], axis=0))
                for f in range(num_fields)
            )
            yield batch
