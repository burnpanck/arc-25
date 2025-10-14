"""k-NN evaluation for learned representations."""

import typing

import attrs
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..vision2.encoder import ARCEncoder
from .dataset import BucketedDataset, MinibatchSizeFunction


@attrs.frozen
class KNNEvaluator:
    """k-NN evaluator for encoder representations.

    Holds a fixed evaluation dataset and computes k-NN classification
    accuracy for given k values.
    """

    dataset: BucketedDataset
    batch_size: MinibatchSizeFunction
    k_values: frozenset[int] = frozenset([1, 3, 5, 10])
    seed: int = 42

    def evaluate(
        self,
        encoder: ARCEncoder,
        *,
        mode: str | None = None,
        with_progress: bool = False,
    ) -> dict[int, float]:
        """Evaluate encoder using k-NN classification.

        Args:
            encoder: Encoder model to evaluate
            mode: Mode to pass to encoder (e.g., "flat" or "split")

        Returns:
            Dictionary with keys like "knn_acc_k=5" mapping to accuracy values
        """
        # Encode all examples once
        embeddings, labels = self._encode_dataset(
            encoder, mode=mode, with_progress=with_progress
        )

        # Compute similarity matrix once (cosine similarity)
        similarity = embeddings @ embeddings.T
        # Mask out self-similarity (set diagonal to -inf for leave-one-out)
        n_examples = len(embeddings)
        similarity = similarity - np.eye(n_examples) * np.inf

        # compute nearest neighbours
        nearest_indices = np.argsort(-similarity, axis=-1)[:, : max(self.k_values)]
        neighbor_labels = labels[nearest_indices]
        same_class = neighbor_labels == labels[:, None]
        n_same_class = np.cumsum(same_class, axis=-1)

        # Compute k-NN accuracy for each k value
        results = {}
        for k in self.k_values:
            correct = n_same_class[:, k] > k // 2
            accuracy = np.mean(correct)
            results[k] = accuracy

        return results

    @staticmethod
    @nnx.pmap(
        axis_name="data",
        in_axes=(nnx.StateAxes({...: None}), 0, 0, None),
        static_broadcasted_argnums=3,
    )
    def _encode(encoder, images, sizes, mode):
        embeddings = encoder(
            images,
            sizes,
            mode=mode,
            deterministic=True,
        )
        embeddings = embeddings.context.as_flat().data
        # shape is currently (..., T, D), with T the number of context tokens
        embeddings = embeddings.reshape(*embeddings.shape[:-2], -1)

        # Normalize embeddings for cosine similarity
        embeddings = embeddings * (
            1 / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        )
        return embeddings

    def _encode_dataset(
        self,
        encoder: nnx.Module,
        *,
        mode: str | None = "flat",
        with_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode all examples from the dataset.

        Returns:
            Tuple of (embeddings, challenge_labels) where:
            - embeddings: [N, D] array of normalized embeddings
            - challenge_labels: [N] array of challenge IDs
        """
        all_embeddings = []
        all_challenge_labels = []

        num_devices = self.batch_size.granularity
        rgen = np.random.default_rng(self.seed)

        # Iterate over all buckets
        for bucket_shape, minibatch_data in sorted(
            self.dataset.buckets.items(), key=lambda kv: kv[0]
        ):
            n_examples = minibatch_data.n_examples
            batch_size = self.batch_size(int(np.prod(bucket_shape)))
            assert batch_size > 0
            assert not batch_size % num_devices
            n_batches = n_examples // batch_size
            assert n_batches > 0
            batches = rgen.choice(
                n_examples,
                size=(n_batches, num_devices, batch_size // num_devices),
                replace=False,
                shuffle=False,
            )

            # Process in batches
            if with_progress:
                import tqdm.auto

                it = tqdm.auto.tqdm(batches, leave=False)
            else:
                it = batches
            for batch in it:
                # Extract batch
                batch_images = minibatch_data.images[batch]
                batch_sizes = minibatch_data.sizes[batch]
                batch_labels = minibatch_data.labels[batch, 0]  # Challenge ID

                # Encode batch
                embeddings = self._encode(
                    encoder,
                    batch_images,
                    batch_sizes,
                    mode,
                )
                embeddings = embeddings.reshape(-1, embeddings.shape[-1])

                all_embeddings.append(jax.copy_to_host_async(embeddings))
                all_challenge_labels.append(batch_labels.ravel())

        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        challenge_labels = np.concatenate(all_challenge_labels, axis=0)

        return embeddings, challenge_labels
