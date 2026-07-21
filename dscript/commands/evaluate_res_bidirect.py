"""
Evaluate a trained interaction_res_bidirect model.
"""

from __future__ import annotations

import argparse

from ..models.interaction_res_bidirect import InteractionInputs
from . import evaluate_res as _evaluate_res

_evaluate_res.InteractionInputs = InteractionInputs


def _predict_bidirect_logit(model, interaction_inputs):
    _, logit12, _ = model.map_predict(interaction_inputs)
    reverse_inputs = InteractionInputs(
        interaction_inputs.z1,
        interaction_inputs.z0,
        embed_foldseek=interaction_inputs.embed_foldseek,
        f0=interaction_inputs.f1,
        f1=interaction_inputs.f0,
        embed_backbone=interaction_inputs.embed_backbone,
        b0=interaction_inputs.b1,
        b1=interaction_inputs.b0,
    )
    _, logit21, _ = model.map_predict(reverse_inputs)
    return 0.5 * (logit12.view(-1).float() + logit21.view(-1).float())


_evaluate_res.predict_logit = _predict_bidirect_logit

EvaluateArguments = _evaluate_res.EvaluateArguments
add_args = _evaluate_res.add_args
main = _evaluate_res.main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
