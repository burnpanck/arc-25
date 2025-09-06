import types

import attrs
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Trainer

from ..dataset import Challenge
from ..dsl.types import Canvas, Color, IOPair
from ..prompts import PromptEncoder, ReasonedSolution


@attrs.frozen(kw_only=True)
class SFTEncoder:
    encoder: PromptEncoder
    model_name: str
    tokenizer: AutoTokenizer
    additional_special_tokens: tuple[str, ...]
    max_length: int = 4096

    _token_to_tag: dict[int, str] = attrs.field(
        default=attrs.Factory(
            lambda self: types.MappingProxyType(
                {
                    self.tokenizer.convert_tokens_to_ids(v): tc + k
                    for tc, ts in {
                        "+": self.encoder.open_tokens,
                        "-": self.encoder.close_tokens,
                    }.items()
                    for k, v in vars(ts).items()
                }
            ),
            takes_self=True,
        )
    )
    _im_tok: dict[str, int] = attrs.field(
        default=attrs.Factory(
            lambda self: types.MappingProxyType(
                {
                    k: self.tokenizer.convert_tokens_to_ids(f"<|im_{k}|>")
                    for k in ["start", "end"]
                }
            ),
            takes_self=True,
        )
    )

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        *,
        encoder: PromptEncoder | None = None,
        encoder_args: dict | None = None,
        **kw,
    ):
        if encoder is None:
            encoder = PromptEncoder(**encoder_args or {})
        else:
            assert encoder_args is None
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        additional_special_tokens = tuple(
            list(encoder.colour_tokens)
            + list(vars(encoder.open_tokens).values())
            + list(vars(encoder.close_tokens).values())
        )
        tok.add_special_tokens(
            dict(additional_special_tokens=additional_special_tokens)
        )
        return cls(
            tokenizer=tok,
            encoder=encoder,
            model_name=model_name,
            additional_special_tokens=additional_special_tokens,
            **kw,
        )

    def build_chat_text(self, chal: Challenge, resp: ReasonedSolution | None = None):
        messages = self.encoder.prepare_chat_messages(chal, resp)
        messages = [dict(role=k, content=v) for k, v in messages.items()]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=resp
            is None,  # False for SFT labels, True for inference
        )

    def tokenize_with_sections(self, text: str):
        tok = self.tokenizer
        enc = tok(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            return_offsets_mapping=True,
        )
        enc = {k: v[0, :] for k, v in enc.items()}
        input_ids = enc.get("input_ids")
        offsets = enc.pop("offset_mapping")

        # 1) Find the assistant turn boundaries: <|im_start|>assistant ... <|im_end|>
        #    We’ll only supervise inside; everything else -> labels = -100
        # Scan all <|im_start|> occurrences and pick the one where next token piece is "assistant"
        # Qwen template goes: <|im_start|>, "assistant", '\n', content..., <|im_end|>
        role_tokens = tok.encode("assistant\n", add_special_tokens=False)
        starts = np.flatnonzero(input_ids == self._im_tok["start"])
        ends = np.flatnonzero(input_ids == self._im_tok["end"])
        assistant_sections = []
        for i in starts:
            s = input_ids[i + 1 :][: len(role_tokens)]
            if np.all(s == role_tokens):
                j = np.searchsorted(ends, i)
                if j >= len(ends):
                    j = len(input_ids)
                else:
                    j = ends[j]
                assistant_sections.append((i, j))

        # 2) Inside assistant: assign tag weights + section content weights.
        # Walk tokens and push/pop when we encounter section tags.
        # Since tags are single tokens, we can index them directly.

        edges = [0]
        src_edges = [0]
        sections = [None]
        all_special_tok = np.array(list(self._token_to_tag.keys()))
        for s, e in assistant_sections:
            open_stack = []  # holds section names currently open
            edges.append(s)
            src_edges.append(offsets[s, 0])
            sections.append(Ellipsis)

            ids = input_ids[s:e]
            pos = np.flatnonzero(np.any(ids[..., None] == all_special_tok, -1))
            for p in pos:
                t = self._token_to_tag[ids[p]]
                if t.startswith("+"):
                    t = t[1:]
                    edges.append(s + p)
                    src_edges.append(offsets[s + p, 0])
                    sections.append(t)
                    open_stack.append(t)
                else:
                    assert t.startswith("-")
                    t = t[1:]
                    assert open_stack[-1] == t
                    edges.append(s + p + 1)
                    src_edges.append(offsets[s + p, 1])
                    open_stack.pop()
                    sections.append(Ellipsis)

            edges.append(e)
            src_edges.append(offsets[e, 1] if e < offsets.shape[0] else len(text))
            sections.append(None)

        edges.append(len(input_ids))
        src_edges.append(len(text))
        return dict(
            edges=np.array(edges),
            src_edges=np.array(src_edges),
            sections=tuple(sections),
            **enc,
        )

    def create_labels_and_weights(
        self,
        inp,
        *,
        weight_map=types.MappingProxyType(
            dict(descr=0.5, rule=0.6, plan=0.8, impl=1.0)  # noqa: B006
        ),
        special_weight=0.2,
        pad_to: int | None = None,
    ):
        tok = self.tokenizer
        toks = inp["input_ids"]
        edg = inp["edges"]
        sec = inp["sections"]
        labels = []
        weights = []
        for s, e, t in zip(edg[:-1], edg[1:], sec):
            n = e - s
            if not n:
                continue
            if t is None and not s:
                # this is the first user section; not supervising the token
                labels.append(np.tile(-100, n))
                weights.append(np.zeros(n))
                continue
            weights.append([special_weight])
            w = weight_map.get(t)
            if w is None:
                labels.append(toks[s : s + 1])
                labels.append(np.tile(-100, n - 1))
                weights.append(np.zeros(n - 1))
            else:
                labels.append(toks[s:e])
                weights.append(np.tile(w, n - 1))
        labels = np.concatenate(labels)
        weights = np.concatenate(weights)
        assert labels.shape == weights.shape == toks.shape
        ret = dict(
            input_ids=toks,
            labels=labels,
            weights=weights,
            attention_mask=inp["attention_mask"],
        )
        if pad_to is not None and (n := pad_to - toks.size) > 0:
            pad_token = tok.convert_tokens_to_ids(
                tok.eos_token if tok.pad_token is None else tok.pad_token
            )
            padding = dict(
                input_ids=pad_token,
                labels=-100,
                weights=0.0,
                attention_mask=0,
            )
            ret = {
                k: np.concatenate([v, np.tile(np.array(padding[k], v.dtype), n)])
                for k, v in ret.items()
            }

        return ret


class WeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, *, return_outputs=False, num_items_in_batch=None
    ):
        # forward first
        outputs = model(
            **{k: v for k, v in inputs.items() if k not in ("labels", "weights")}
        )
        logits = outputs.logits  # on the model’s last shard device
        dev = logits.device

        # fetch targets/weights and move to *logits* device (not labels.device!)
        labels = inputs["labels"]
        weights = inputs["weights"]
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, dtype=torch.float32)
        labels = labels.to(dev)
        weights = weights.to(dev, dtype=torch.float32)

        # causal shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_wts = weights[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss_tok = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view_as(shift_labels)

        active = (shift_labels != -100).float()
        weighted = loss_tok * shift_wts * active
        denom = (shift_wts * active).sum().clamp_min(1e-6)
        loss = weighted.sum() / denom

        return (loss, outputs) if return_outputs else loss


class CollatorWithWeights:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self._pad_token_id = tokenizer.convert_tokens_to_ids(
            self.tok.eos_token if self.tok.pad_token is None else self.tok.pad_token
        )

    def __call__(self, features):
        padding_values = dict(
            input_ids=self._pad_token_id,
            attention_mask=0,
            labels=-100,
            weights=0.0,
        )

        ret = {}
        m = self.pad_to_multiple_of
        for k, pad_val in padding_values.items():
            # TODO: we should probably do the padding in one go
            padded = pad_sequence(
                [f[k] for f in features], batch_first=True, padding_value=pad_val
            )
            if self.pad_to_multiple_of:
                L = padded.size(-1)
                if L % m:
                    padded = F.pad(padded, (0, m - (L % m)), value=pad_val)
            ret[k] = padded
        return ret
