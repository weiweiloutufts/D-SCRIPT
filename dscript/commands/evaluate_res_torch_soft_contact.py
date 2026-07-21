"""
Evaluate a trained torch-soft-contact residual interaction model.
"""

from __future__ import annotations

import argparse

from ..models.interaction_res_torch_soft_contact_not import InteractionInputs
from . import evaluate_res as _evaluate_res

EvaluateArguments = _evaluate_res.EvaluateArguments
add_args = _evaluate_res.add_args


def main(args):
    _evaluate_res.InteractionInputs = InteractionInputs
    _evaluate_res.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
