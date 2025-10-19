import dataclasses
import lzma
import typing
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

import attrs
import cbor2
import jaxtyping as jt
import numpy as np

from .. import serialisation
from ..dataset import Image


@attrs.frozen
class ImageExample:
    challenge: str
    example_idx: int
    example_type: Literal["input", "output"]
    image: Image


@attrs.frozen
class ImagesDataset:
    examples: tuple[ImageExample, ...]
    challenges: frozenset[str]
    max_size: tuple[int, int] = (30, 30)

    @classmethod
    def load_compressed_cbor(cls, fpath: Path):
        with lzma.LZMAFile(fpath, "rb") as fh:
            src_data = serialisation.deserialise(cbor2.load(fh))
        max_size = np.r_[0, 0]
        examples = []
        for k, v in src_data.items():
            for i, iop in enumerate(v):
                for kk in ["input", "output"]:
                    vv = getattr(iop, kk)
                    sh = np.array(vv.shape)
                    max_size = np.maximum(max_size, sh)
                    examples.append(
                        ImageExample(
                            challenge=k,
                            example_idx=i,
                            example_type=kk,
                            image=vv,
                        )
                    )
        return cls(
            examples=tuple(examples),
            challenges=frozenset(src_data),
            max_size=tuple(int(v) for v in max_size),
        )

    def size_counts(self, max_size: tuple[int, int] | None = None) -> np.ndarray:
        if max_size is None:
            max_size = self.max_size

        ret = np.zeros(tuple(max_size), int)
        for im in self.examples:
            vv = im.image
            if any(s > m for s, m in zip(vv.shape, max_size)):
                continue
            h, w = vv.shape
            ret[h - 1, w - 1] += 1
        return ret

    def split_by_challenge(
        self, rgen: np.random.Generator, n_min: int = 0, fraction: float = 0
    ) -> tuple[Self, Self]:
        """Split dataset by (challenge, example_idx) pairs.

        For each challenge, ensures at least max(n_min, fraction * total_pairs) I/O pairs
        go to the first split, with the rest going to the second split.

        Args:
            rgen: Random number generator for shuffling
            n_min: Minimum number of I/O pairs per challenge in first split
            fraction: Minimum fraction of I/O pairs per challenge in first split

        Returns:
            Tuple of (first_split, second_split) datasets
        """
        from collections import defaultdict

        # Group examples by (challenge, example_idx)
        pairs = defaultdict(list)
        for ex in self.examples:
            pairs[ex.challenge, ex.example_idx].append(ex)

        # Group pairs by challenge
        by_challenge = defaultdict(list)
        for (challenge, _example_idx), pair in pairs.items():
            by_challenge[challenge].append(pair)

        # Split examples
        ret = ([], [])

        for pairs in by_challenge.values():
            # Shuffle pairs for this challenge
            pairs_shuffled = np.array(pairs, object)
            rgen.shuffle(pairs_shuffled, axis=0)
            examples = pairs_shuffled.ravel()

            # Calculate how many pairs go to first split
            n_examples = len(examples)
            n_first = max(n_min, int(np.ceil(fraction * n_examples)))
            n_first = min(n_first, n_examples // 2)

            # Assign to splits
            for target, part in zip(ret, [examples[:n_first], examples[n_first:]]):
                target.extend(part)

        return tuple(
            type(self)(
                examples=tuple(ex),
                challenges=self.challenges,
                max_size=self.max_size,
            )
            for ex in ret
        )


@attrs.frozen
class MiniBatchData:
    images: jt.Int8[np.ndarray, " B Y X"]
    masks: jt.Bool[np.ndarray, " B Y X"]
    sizes: jt.Int[np.ndarray, " B 2"]

    # (challenge, i/o-pair, i or o)
    labels: jt.Int[np.ndarray, " B 3"]
    transpose: jt.Bool[np.ndarray, " B"]
    weight: jt.Float[np.ndarray, " B"]

    @property
    def n_examples(self) -> int:
        return len(self.images)


@attrs.frozen(kw_only=True)
class BucketedDataset:
    buckets: MappingProxyType[tuple[int, int], MiniBatchData]
    challenges: tuple[str, ...]
    allow_transpose: bool = True

    @classmethod
    def make(
        cls,
        dataset: ImagesDataset,
        bucket_shapes: set[tuple[int, int]],
        allow_transpose: bool = True,
    ) -> Self:
        # Create challenge_to_label mapping
        challenges = tuple(sorted(dataset.challenges))
        challenge_to_label = {ch: i for i, ch in enumerate(challenges)}

        # Group examples by bucket
        bucket_examples: dict[tuple[int, int], list[ImageExample]] = {
            shape: [] for shape in bucket_shapes
        }

        bucket_shapes = sorted(
            bucket_shapes,
            key=lambda shape: (shape[0] * shape[1], abs(shape[0] - shape[1])),
        )

        for ex in dataset.examples:
            img = ex.image
            h, w = img.shape

            # If allow_transpose and image is tall, transpose to make it wide
            if allow_transpose and h > w:
                h, w = w, h

            # Find smallest bucket that fits (tie-break by squareness)
            def fits(bucket_shape: tuple[int, int]) -> bool:
                bh, bw = bucket_shape
                return h <= bh and w <= bw

            fitting_buckets = [shape for shape in bucket_shapes if fits(shape)]
            if not fitting_buckets:
                # discard images that won't fit;
                continue

            # Sort by area, then by squareness (minimize |bh - bw|)
            best_bucket = fitting_buckets[0]

            bucket_examples[best_bucket].append(ex)

        # Build MiniBatchData for each bucket
        buckets_dict = {}
        for bucket_shape, examples_list in bucket_examples.items():
            if not examples_list:
                continue

            height, width = bucket_shape
            n_examples = len(examples_list)

            # Create arrays for all examples in this bucket
            images = np.zeros((n_examples, height, width), dtype=np.int8)
            masks = np.zeros((n_examples, height, width), dtype=bool)
            sizes = np.zeros((n_examples, 2), dtype=int)
            labels = np.zeros((n_examples, 3), dtype=int)
            transpose_flags = np.zeros(n_examples, dtype=bool)

            for i, ex in enumerate(examples_list):
                img = ex.image
                h, w = img.shape
                if allow_transpose and h > w:
                    h, w = w, h
                    img = dataclasses.replace(img, _data=img._data.T)
                    transposed = True
                else:
                    transposed = False

                images[i, :h, :w] = img._data
                masks[i, :h, :w] = True
                sizes[i] = [h, w]

                challenge_label = challenge_to_label[ex.challenge]
                example_type_label = ["input", "output"].index(ex.example_type)
                labels[i] = (challenge_label, ex.example_idx, example_type_label)

                transpose_flags[i] = transposed

            # Weights placeholder - will be set by batch_spec later based on actual sizes
            weights = np.ones(n_examples, dtype=float)

            # Create MiniBatchData containing all examples for this bucket
            minibatch_data = MiniBatchData(
                images=images,
                masks=masks,
                sizes=sizes,
                labels=labels,
                transpose=transpose_flags,
                weight=weights,
            )

            buckets_dict[bucket_shape] = minibatch_data

        return cls(
            buckets=MappingProxyType(buckets_dict),
            allow_transpose=allow_transpose,
            challenges=challenges,
        )


@attrs.frozen
class MinibatchSizeFunction:
    # measured in terms of reference images
    reference_minibatch_size: int | float
    reference_image_size: int | float = 30
    # the memory cost is estimated as proportional to `base_cost + image_area` in units of cells
    base_cost: int | float = 30
    # the result is a multiple of the granularity
    granularity: int = 1

    def __call__(self, area: int) -> int:
        cost = self.base_cost + area
        ref_cost = self.base_cost + self.reference_image_size**2
        ret = int(self.reference_minibatch_size * ref_cost / cost)
        ret -= ret % self.granularity
        return ret


@attrs.frozen
class BatchSpec:
    # measured in terms of reference images
    target_batch_weight: int | float
    reference_image_size: int | float = 30
    #
    area_weight_exponent: float = 0

    def __call__(self, area: int) -> float:
        if not self.area_weight_exponent:
            return 1
        return (area / self.reference_image_size**2) ** self.area_weight_exponent


@attrs.mutable
class BucketState:
    bucket_shape: tuple[int, int] = attrs.field(on_setattr=attrs.setters.frozen)
    examples: MiniBatchData = attrs.field(on_setattr=attrs.setters.frozen)
    minibatch_size: int = attrs.field(on_setattr=attrs.setters.frozen)

    # the remaining number of examples from this bucket, this epoch; can be negative
    # if we already used a number of examples from the next epoch to fill a batch
    remaining: int = attrs.field(
        default=attrs.Factory(lambda self: self.examples.n_examples, takes_self=True)
    )
    # contains indices into `examples`, with `shuffle_buffer[:remaining]` still available
    shuffle_buffer: np.ndarray = attrs.field(
        default=attrs.Factory(
            lambda self: np.arange(self.examples.n_examples, dtype=int), takes_self=True
        )
    )

    def sample(self, rng: np.random.Generator) -> MiniBatchData:
        N = self.examples.n_examples
        rem = self.remaining
        nB = self.minibatch_size
        buf = self.shuffle_buffer
        if rem < nB:
            i = rem + rng.choice(a=N - rem, size=nB - rem, replace=False, shuffle=False)
            choice = np.r_[buf[:rem], buf[i]]
            # move to the back of the queue only those selected from the new batch
            for k, j in enumerate(reversed(i), 1):
                k = N - k
                buf[[j, k]] = buf[[k, j]]
        else:
            i = rng.choice(a=rem, size=nB, replace=False, shuffle=False)
            choice = buf[i]
            # move the chosen to the back of the queue
            for k, j in enumerate(reversed(i), 1):
                k = rem - k
                buf[[j, k]] = buf[[k, j]]

        self.remaining -= nB

        return MiniBatchData(
            **{k: v[choice] for k, v in attrs.asdict(self.examples).items()}
        )


@attrs.frozen
class BatchData:
    # --- status data at the start of this batch
    # global step count
    step: int
    # global epoch count
    epoch: int
    # fractional epoch progress
    epoch_progress: float
    # accumulated example weight prior to this batch
    accumulated_weight: float

    # should roughly be equal to `batch_size`
    total_weight: float
    total_examples: int
    minibatches: tuple[MiniBatchData, ...]


@attrs.mutable
class BucketedCollator:
    dataset: BucketedDataset = attrs.field(on_setattr=attrs.setters.frozen)
    batch_spec: BatchSpec = attrs.field(on_setattr=attrs.setters.frozen)
    buckets: MappingProxyType[tuple[int, int], BucketState] = attrs.field(
        on_setattr=attrs.setters.frozen
    )
    seed: int = attrs.field(on_setattr=attrs.setters.frozen)
    # epoch length in number of examples
    total_examples: int = attrs.field(on_setattr=attrs.setters.frozen)
    total_example_weight: float = attrs.field(on_setattr=attrs.setters.frozen)
    # global step count
    step: int = 0
    # global epoch count
    epoch: int = 0
    # global accumulated weight
    accumulated_weight: float = 0
    # example count within the epoch; can go slightly above total_examples due to minibatch completion
    example_in_epoch: int = 0
    # cumulated example_weight within the epoch; can go slightly above total_example_weight due to minibatch completion
    weight_in_epoch: float = 0

    rng: np.random.Generator = attrs.field(
        default=attrs.Factory(
            lambda self: np.random.default_rng(self.seed), takes_self=True
        )
    )

    @classmethod
    def make(
        cls,
        dataset: BucketedDataset,
        batch_spec: BatchSpec,
        minibatch_size: MinibatchSizeFunction,
        seed: int,
    ) -> Self:
        buckets_dict = {}
        total_examples = 0
        total_example_weight = 0.0

        for bucket_shape, minibatch_data in dataset.buckets.items():
            height, width = bucket_shape
            bucket_area = height * width

            # Calculate minibatch size based on bucket area
            mbs = minibatch_size(bucket_area)

            n_examples = len(minibatch_data.images)

            # Calculate weight for each example based on its actual size
            weights = batch_spec(np.prod(minibatch_data.sizes, axis=-1))

            # Update weights in the MiniBatchData
            weighted_minibatch_data = attrs.evolve(
                minibatch_data,
                weight=weights,
            )

            # Create BucketState
            bucket_state = BucketState(
                bucket_shape=bucket_shape,
                examples=weighted_minibatch_data,
                minibatch_size=mbs,
            )

            buckets_dict[bucket_shape] = bucket_state

            total_examples += n_examples
            total_example_weight += float(weights.sum())

        return cls(
            dataset=dataset,
            batch_spec=batch_spec,
            buckets=MappingProxyType(buckets_dict),
            seed=seed,
            total_examples=total_examples,
            total_example_weight=total_example_weight,
        )

    def generate(self) -> typing.Iterator[BatchData]:
        target_batch_weight = self.batch_spec.target_batch_weight
        minibatches = []
        total_weight = 0.0
        total_examples = 0

        # Capture epoch info at start of batch
        start_epoch = self.epoch

        while True:
            # Check if we need to start a new epoch
            if not any(bucket.remaining > 0 for bucket in self.buckets.values()):
                # Start new epoch
                self.epoch += 1
                self.example_in_epoch -= self.total_examples
                self.weight_in_epoch -= self.total_example_weight
                for bucket in self.buckets.values():
                    N = len(bucket.examples.images)
                    bucket.remaining += N

            # Select a bucket with probability proportional to remaining images
            available_buckets = [
                bucket for bucket in self.buckets.values() if bucket.remaining > 0
            ]

            assert available_buckets

            remainings = np.array(
                [
                    bucket.remaining / bucket.minibatch_size
                    for bucket in available_buckets
                ]
            )
            probs = remainings / remainings.sum()

            # Select a bucket
            bucket_idx = self.rng.choice(len(available_buckets), p=probs)
            bucket = available_buckets[bucket_idx]

            # Sample a minibatch from that bucket
            minibatch = bucket.sample(self.rng)

            # Sum the weights in the minibatch
            minibatch_weight = float(minibatch.weight.sum())
            n_examples = len(minibatch.images)
            assert n_examples == bucket.minibatch_size

            # Check if that minibatch belongs to this or the next batch
            if (
                minibatches
                and (total_weight + minibatch_weight) * total_weight
                > target_batch_weight**2
            ):
                # Update counters for the batch we're about to yield
                self.example_in_epoch += total_examples
                self.weight_in_epoch += total_weight
                self.accumulated_weight += total_weight
                self.step += 1

                # Prepare batch data
                # step and accumulated_weight reflect state AFTER this batch
                # epoch and epoch_progress reflect state at START of this batch
                batch_data = BatchData(
                    step=self.step,
                    epoch=start_epoch,
                    epoch_progress=self.weight_in_epoch / self.total_example_weight,
                    accumulated_weight=self.accumulated_weight,
                    total_weight=total_weight,
                    total_examples=total_examples,
                    minibatches=tuple(minibatches),
                )

                yield batch_data

                # Prepare next batch - capture new epoch info
                start_epoch = self.epoch
                minibatches = []
                total_weight = 0.0
                total_examples = 0

            # Add to batch
            minibatches.append(minibatch)
            total_weight += minibatch_weight
            total_examples += n_examples
