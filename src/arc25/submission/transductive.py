import contextlib
import itertools
import json
import lzma
import os
import sys
import time
import typing
from pathlib import Path
from types import SimpleNamespace

import anyio
import attrs
import click
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import optax
import tqdm.auto
from flax import nnx

from .. import dataset as challenge_dataset
from ..lib.click_tools import attrs_to_click_options
from ..serialisation import serialisable, serialise
from ..training import arc_solver as solver_trainer
from ..training import cli, dataset, saving
from ..vision2 import arc_solver


class STrainState(solver_trainer.TrainState):
    train_filter: typing.ClassVar = nnx.filterlib.All(
        nnx.Param, nnx.PathContains("latent_program_embeddings")
    )


num_devices = 4
num_solution_attempts = 8


def serialise_msgpack_file(outfile: Path, src: typing.Any):
    outfile = Path(outfile)
    reduced = serialise(src, reduce=saving.reduce_jax)
    serialised = msgpack.dumps(reduced)
    with contextlib.ExitStack() as stack:
        match outfile.suffix:
            case ".msgpack":
                fh = stack.enter_context(open(outfile, "wb"))
            case ".xz":
                fh = stack.enter_context(lzma.LZMAFile(outfile, "wb"))
            case _:
                raise KeyError(f"Unsupported suffix {outfile.suffix}")
        fh.write(serialised)


@serialisable
@attrs.frozen(kw_only=True)
class Config:
    """Unified training configuration for both MAE and ArcSolver."""

    challenges_input_file: Path = Path(
        "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
    )
    model_weights_file: Path
    output_file: Path = Path("/kaggle/working/submission.json")
    prediction_batch_size: int = 16 * num_devices
    latent_program_init_scale: float = 0.2

    model: cli.ModelSelection = cli.ModelSelection(
        type="arc-solver",
        config="small",
        dtype="bfloat16",
    )

    size_bins: frozenset[int] = frozenset([30])

    # WARNING: These defaults are currently not respected by `attrs_to_click_options`!
    trainer: solver_trainer.ArcSolverConfig = solver_trainer.ArcSolverConfig(
        batch_size=128,
        base_cell_cost=0,
        minibatch_size=64 * num_devices,
        eval_batch_size=None,  # 32 * num_devices // num_solution_attempts,
        learning_rate=1e-3,
        max_num_epochs=100,
        warmup_steps=16,
        checkpoint_every_steps=None,
        eval_every_ref_batch=None,
        num_solution_attempts=num_solution_attempts,
        reference_image_size=15,
        ref_batch=256,
        max_num_ref_batches=None,
        mode="flat",
        remat=True,
        unroll=None,
        loss_focus=0,
    )


@click.command()
@attrs_to_click_options
def solve(config: Config):
    tstart = time.monotonic()

    print(
        "Solving ARC Prize 2025 (or at least attempting to ☺️) using transductive approach"
    )
    print(f"Config: {config}")
    sys.stdout.flush()

    num_devices = jax.local_device_count()
    assert not config.prediction_batch_size % num_devices

    trainer_config = config.trainer

    ts = time.monotonic()
    print("\n*** Load data")
    input_data = Path(config.challenges_input_file)
    print(f"Loading from {input_data}")
    main_dataset = anyio.run(
        lambda: challenge_dataset.Dataset.load_from_json(
            root=input_data.parent,
            challenges=input_data.name,
            id="main",
        )
    )

    max_size = np.r_[0, 0]
    datasets = dict(train=[], test=[])
    for k, v in main_dataset.challenges.items():
        for typ, dst in datasets.items():
            if typ == "train" and any(
                iop.input.shape != iop.output.shape for iop in getattr(v, typ)
            ):
                continue
            for i, iop in enumerate(getattr(v, typ)):
                for kk in ["input", "output"]:
                    if typ == "test" and kk == "output":
                        continue
                    img = getattr(iop, kk)
                    ex = dataset.ImageExample(
                        challenge=k,
                        example_idx=i,
                        example_type=kk,
                        image=img,
                    )
                    sh = np.array(img.shape)
                    max_size = np.maximum(max_size, sh)
                    dst.append(ex)
    challenges = frozenset(main_dataset.challenges)
    challenge_order = tuple(sorted(challenges))
    datasets = SimpleNamespace(
        **{
            k: dataset.ImagesDataset(
                examples=tuple(v),
                challenges=challenges,
                max_size=tuple(int(v) for v in max_size),
            )
            for k, v in datasets.items()
        }
    )

    te = time.monotonic()
    print(f"Data loading complete, took {te-ts:.1f} s")
    sys.stdout.flush()
    ts = te

    print("\n*** Build model")
    sys.stdout.flush()
    solver = arc_solver.ARCSolver(
        **arc_solver.configs[config.model.config],
        num_latent_programs=len(challenge_order) * config.trainer.num_solution_attempts,
        dtype=getattr(jnp, config.model.dtype),
        rngs=nnx.Rngs(42),
    )
    te = time.monotonic()
    print(f"Model building complete, took {te-ts:.1f} s")
    ts = te

    chkp_path = Path(config.model_weights_file)
    print(f"Now loading model weights from {chkp_path}")
    checkpoint_data = saving.load_model(chkp_path)
    solver_checkpoint = checkpoint_data.state.model
    del solver_checkpoint["latent_program_embeddings"]
    nnx.update(solver, solver_checkpoint)
    rgen = np.random.default_rng(seed=trainer_config.seed)
    solver.latent_program_embeddings.value = (
        config.latent_program_init_scale
        * rgen.normal(size=solver.latent_program_embeddings.shape)
    )

    te = time.monotonic()
    print(f"Loading weights complete, took {te-ts:.1f} s")
    sys.stdout.flush()
    ts = te

    print("\n*** Identify latent programs")
    input_ds, output_ds = datasets.train.split_input_output()

    bucket_shapes = tuple(
        sorted(set(itertools.product(config.size_bins, config.size_bins)))
    )

    (training_ds,) = [
        dataset.BucketedDataset.make(
            ds,
            bucket_shapes,
            challenges=challenge_order,
        )
        for ds in [output_ds]
    ]

    # Create collator with proper seed and granularity
    minibatch_size_fn = dataset.MinibatchSizeFunction(
        reference_minibatch_size=trainer_config.minibatch_size,
        reference_image_size=trainer_config.reference_image_size,
        base_cost=trainer_config.base_cell_cost,
        granularity=num_devices,
    )

    batch_spec = dataset.BatchSpec(
        target_batch_weight=trainer_config.batch_size,
        reference_image_size=trainer_config.reference_image_size,
        area_weight_exponent=None,
    )

    collator = dataset.BucketedCollator.make(
        dataset=training_ds,
        batch_spec=batch_spec,
        minibatch_size=minibatch_size_fn,
        seed=trainer_config.seed,
    )

    input_src = dataset.OnDemandBucketDataset(
        input_ds,
        bucket_shapes=bucket_shapes,
        challenges=challenge_order,
        weight_fun=lambda area: None,
    )

    rngs = nnx.Rngs(trainer_config.seed)
    trainer = solver_trainer.ArcSolverTrainer.make(
        config=trainer_config,
        model=solver,
        collator=collator,
        inputs_src=input_src,
        num_devices=num_devices,
        rngs=rngs,
        eval_dataset=None,
        with_progress_bars=True,
    )

    trainer.train_state = STrainState.make(solver, trainer_config, rngs=rngs)

    te = time.monotonic()
    print(f"Trainer is set up, took {te-ts:.1f} s")
    ts = te

    print(
        f"Doing gradient descent for {trainer_config.max_num_epochs} epochs to identify latent programs"
    )
    res = trainer.run_main()

    training_hist_file_name = config.output_file.with_name("training-hist.msgpack.xz")
    print(f"Storing training history to {training_hist_file_name}")
    serialise_msgpack_file(
        training_hist_file_name,
        res,
    )

    latent_pgm_file_name = config.output_file.with_name("latent-pgms.msgpack.xz")
    print(f"Storing latent programs to {latent_pgm_file_name}")
    lpe = np.asarray(solver.latent_program_embeddings)
    nsa = trainer_config.num_solution_attempts

    serialise_msgpack_file(
        latent_pgm_file_name,
        dict(
            challenges=challenge_order,
            latent_program_embeddings=lpe.reshape(-1, nsa, *lpe.shape[1:]),
        ),
    )

    te = time.monotonic()
    print(f"Training complete, final loss {res[-1]['loss']:.3f}, took {te-ts:.1f} s")
    sys.stdout.flush()
    ts = te

    print("\n*** Compute predictions")
    sys.stdout.flush()
    print("Prepare model inputs")
    all_inputs = []
    for challenge in main_dataset.challenges.values():
        for subset in ["train", "test"]:
            for i, ex in enumerate(getattr(challenge, subset)):
                img = ex.input
                h, w = img.shape
                input = np.zeros((30, 30), "i1")
                input[:h, :w] = img._data
                size = np.array([h, w])
                all_inputs.append(
                    SimpleNamespace(
                        challenge=challenge,
                        example_type=subset,
                        example_idx=i,
                        input=input,
                        size=size,
                        challenge_idx=challenge_order.index(challenge.id),
                    )
                )

    te = time.monotonic()
    print(f"Model inputs prepared (total {len(all_inputs)} images), took {te-ts:.1f} s")
    ts = te

    print(f"Compute predictions in batches of {config.prediction_batch_size}")
    model_kw = dict(
        mode=trainer_config.mode,
        remat=False,
        unroll=trainer_config.unroll,
    )

    @nnx.jit
    def predict(solver, inputs, sizes, challenge_idx):
        embeddings = solver.encoder(
            inputs,
            sizes,
            **model_kw,
        )
        data = jax.tree.map(
            lambda a: jnp.tile(a[:, None, ...], (1, nsa) + (1,) * (a.ndim - 1)),
            dict(embeddings=embeddings, sizes=sizes, challenge_idx=challenge_idx),
        )
        embeddings = data["embeddings"]
        sizes = data["sizes"]
        latent_program_idx = data["challenge_idx"] * nsa
        latent_program_idx += np.arange(nsa)[
            None, :, *(None,) * (latent_program_idx.ndim - 2)
        ]

        logits = solver.decode(
            embeddings,
            output_size=sizes,
            latent_program_idx=latent_program_idx,
            **model_kw,
        ).astype(jnp.float32)

        return logits

    mesh = jax.make_mesh(
        (num_devices,),
        ("batch",),
        axis_types=(jax.sharding.AxisType.Auto,),
    )

    def reshard(a, *args):
        return jax.device_put(
            a,
            jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args)),
        )

    # reshard solver onto the explicit mesh
    model_graph, model_state = nnx.split(solver)
    model_state = jax.tree.map(lambda a: reshard(a), model_state)
    resharded_solver = nnx.merge(model_graph, model_state)

    solutions = {}

    def handle(batch):
        inputs = reshard(np.array([inp.input for inp in batch]), "batch")
        sizes = reshard(np.array([inp.size for inp in batch]), "batch")
        challenge_idx = reshard(np.array([inp.challenge_idx for inp in batch]), "batch")
        pred = predict(resharded_solver, inputs, sizes, challenge_idx)
        pred = jax.copy_to_host_async(pred)
        for i, inp in enumerate(batch):
            chal = inp.challenge
            solution = solutions.setdefault(
                chal.id,
                SimpleNamespace(
                    challenge=chal,
                    train_logits=[None] * len(chal.train),
                    test_logits=[None] * len(chal.test),
                ),
            )
            getattr(solution, f"{inp.example_type}_logits")[inp.example_idx] = pred[i]

    batch_size = config.prediction_batch_size
    remainder = None
    for start in tqdm.auto.trange(0, len(all_inputs), batch_size):
        batch = all_inputs[start : start + batch_size]
        n = len(batch)
        if n < batch_size:
            n -= n % num_devices
            batch, remainder = batch[:n], batch[n:]
        if batch:
            handle(batch)

    if remainder:
        # single-device mesh to handle remainder
        mesh = jax.make_mesh(
            (1,),
            ("batch",),
            axis_types=(jax.sharding.AxisType.Auto,),
        )
        model_graph, model_state = nnx.split(solver)
        model_state = jax.tree.map(lambda a: reshard(a), model_state)
        resharded_solver = nnx.merge(model_graph, model_state)
        handle(remainder)

    te = time.monotonic()
    print(f"Predictions complete, took {te-ts:.1f} s")
    ts = te

    print("Evaluating predictions")
    for solution in solutions.values():
        challenge = solution.challenge

        Nt = len(solution.train_logits)

        logits = np.array(solution.train_logits)
        outputs = np.zeros((Nt, 1, 30, 30), "i1")
        output_masks = np.zeros((Nt, 1, 30, 30), bool)
        output_area = np.zeros((Nt, 1), int)
        output_size_match = np.zeros((Nt, 1), bool)
        for i, ex in enumerate(challenge.train):
            h, w = ex.output.shape
            m = ex.output.shape == ex.input.shape
            outputs[i, :, :h, :w] = ex.output._data
            output_size_match[i, :] = m
            output_masks[i, :, :h, :w] = True if m else False
            output_area[i, :] = h * w

        cell_crossentropy = np.array(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=outputs, axis=-1
            )
        )

        cell_weights = output_masks / output_area[..., None, None]
        # Mask to valid output regions and weight by pre-normalized cell weights
        pair_crossentropy = (
            np.where(output_masks, cell_crossentropy * cell_weights, 0)
            .sum(axis=(-2, -1))
            .mean(0)
        )

        # Per-cell accuracy
        predictions = np.argmax(logits, axis=-1)
        cell_correct = predictions == outputs
        cell_accuracy = (
            np.where(cell_correct & output_masks, cell_weights, 0)
            .astype(jnp.float32)
            .sum(axis=(-2, -1))
            .mean(0)
        )

        # Per-pair accuracy (all cells in output must be correct)
        pair_accuracy = (
            (
                (
                    # Padding doesn't count against accuracy
                    cell_correct
                    | ~output_masks
                ).all(axis=(-2, -1))
                & output_size_match
            )
            .astype(jnp.float32)
            .mean(0)
        )

        # shape (example, latent, Y, X, C)
        test_logits = np.array(solution.test_logits)
        test_probs = np.array(jax.nn.softmax(test_logits, axis=-1))
        # shape (example, latent, Y, X)
        test_predictions = np.argmax(test_logits, axis=-1)

        best_guess_idx = np.argmin(pair_crossentropy)
        best_guess_pred = test_predictions[:, best_guess_idx, :, :]

        voting_weight = np.array(jax.nn.softmax(-pair_crossentropy))
        voted_pred = np.argmax(
            (voting_weight[None, :, None, None, None] * test_probs).sum(1), axis=-1
        )

        final_pred = np.concatenate(
            [best_guess_pred[:, None, :, :], voted_pred[:, None, :, :]], axis=1
        )
        final_pred = tuple(
            pred[:, :h, :w]
            for pred, (h, w) in zip(
                final_pred, (ex.input.shape for ex in challenge.test)
            )
        )

        results = dict(
            pair_accuracy=pair_accuracy,
            pair_crossentropy=pair_crossentropy,
            cell_accuracy=cell_accuracy,
            train_predictions=predictions,
            test_predictions=test_predictions,
            final_predictions=final_pred,
        )
        for k, v in results.items():
            setattr(solution, k, v)

    te = time.monotonic()
    print(f"Evaluation complete, took {te-ts:.1f} s")
    sys.stdout.flush()
    ts = te

    print("\n*** Writing submission file")
    sys.stdout.flush()
    output_data = {}
    for sol in solutions.values():
        output_data[sol.challenge.id] = [
            {
                f"attempt_{i}": [[int(v) for v in row] for row in attempt]
                for i, attempt in enumerate(pred, 1)
            }
            for pred in sol.final_predictions
        ]
    with open(config.output_file, "wt") as fh:
        json.dump(output_data, fh)

    print(f"Output file: {config.output_file}")

    full_results_output = config.output_file.with_name(
        "solutions.msgpack.xz",
    )
    print(f"Writing detailed solutions to: {full_results_output}")
    serialise_msgpack_file(
        full_results_output,
        {
            k: {
                kk: vv.id if isinstance(vv, challenge_dataset.Challenge) else vv
                for kk, vv in vars(v).items()
            }
            for k, v in solutions.items()
        },
    )

    tfinal = time.monotonic()
    print(f"All done; total time {(tfinal-tstart)/60:.1f} mins")
    sys.stdout.flush()


if __name__ == "__main__":
    os.environ["EPATH_USE_TF"] = "false"

    solve()
