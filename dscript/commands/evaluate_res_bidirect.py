"""
Evaluate a trained interaction_res_bidirect model.
"""

from __future__ import annotations

import argparse

from ..models.interaction_res_bidirect import InteractionInputs
from . import evaluate_res as _evaluate_res

_evaluate_res.InteractionInputs = InteractionInputs

EvaluateArguments = _evaluate_res.EvaluateArguments
add_args = _evaluate_res.add_args
main = _evaluate_res.main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
