"""k-NN evaluation for learned representations."""

import typing
from contextlib import ExitStack

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
        similarity = np.where(np.eye(n_examples, dtype=bool), -np.inf, similarity)

        # compute nearest neighbours
        nearest_indices = np.argsort(-similarity, axis=-1)[:, : max(self.k_values)]
        neighbor_labels = labels[nearest_indices]
        is_same = neighbor_labels == labels[:, None, :]
        is_same[..., 1] &= is_same[..., 0]
        n_match = np.cumsum(is_same, axis=1)

        # Compute k-NN accuracy for each k value
        res_challenge = {}
        res_candt = {}
        for k in self.k_values:
            correct = n_match[:, k - 1, :] > k // 2
            achal, acandt = np.mean(correct, axis=0)
            res_challenge[k] = achal
            res_candt[k] = acandt

        return dict(
            challenge=res_challenge,
            candt=res_candt,
        )

    @staticmethod
    @nnx.pmap(
        axis_name="data",
        in_axes=(nnx.StateAxes({...: None}), 0, 0, None),
        static_broadcasted_argnums=3,
    )
    def _encode(encoder, images, sizes, mode):
        print(f"Tracing KNNEvaluator._encode for shape {images.shape}")

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

        with ExitStack() as stack:
            if with_progress:
                n_batches_tot = 0
                for bucket_shape, minibatch_data in self.dataset.buckets.items():
                    n_examples = minibatch_data.n_examples
                    batch_size = self.batch_size(int(np.prod(bucket_shape)))
                    assert batch_size > 0
                    assert not batch_size % num_devices
                    n_batches_tot += n_examples // batch_size
                    if (n_examples - n_examples % num_devices) % batch_size:
                        # there are extra examples remaining; these will get their own batch size
                        n_batches_tot += 1

                import tqdm.auto

                pbar = stack.enter_context(
                    tqdm.auto.tqdm(total=n_batches_tot, leave=False)
                )
            else:
                pbar = None

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
                seq = rgen.permutation(n_examples)
                n_rem = (n_examples - n_examples % num_devices) % batch_size
                if n_rem:
                    batches = [seq[:n_rem].reshape(num_devices, -1)]
                    seq = seq[n_rem:]
                else:
                    batches = []
                batches.extend(
                    seq[: n_batches * batch_size].reshape(n_batches, num_devices, -1)
                )

                pbar.set_postfix(
                    bucket_shape=bucket_shape, bucket_progress=f"0/{len(batches)}"
                )

                # Process in batches
                for i, batch in enumerate(batches):
                    # Extract batch
                    batch_images = minibatch_data.images[batch]
                    batch_sizes = minibatch_data.sizes[batch]
                    batch_labels = minibatch_data.labels[batch, :][
                        ..., [0, 2]
                    ]  # (Challenge ID, Image Type)

                    # Encode batch
                    embeddings = self._encode(
                        encoder,
                        batch_images,
                        batch_sizes,
                        mode,
                    )
                    embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                    batch_labels = batch_labels.reshape(-1, 2)

                    all_embeddings.append(jax.copy_to_host_async(embeddings))
                    all_challenge_labels.append(batch_labels)

                    if pbar:
                        pbar.update()
                        pbar.set_postfix(
                            bucket_shape=bucket_shape,
                            bucket_progress=f"{i+1}/{len(batches)}",
                        )

            # Concatenate all batches
            embeddings = np.concatenate(all_embeddings, axis=0)
            challenge_labels = np.concatenate(all_challenge_labels, axis=0)

        return embeddings, challenge_labels
