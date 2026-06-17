"""
D-SCRIPT: Structure Aware PPI Prediction
"""

import argparse
import sys

from .commands import (
    embed,
    evaluate,
    extract_3di,
    predict_bipartite,
    predict_block,
    predict_serial,
    train,
    train_res_bidirect,
    train_res_enc_auxi,
    train_res_enc_fs_auxi,
    train_res_enc_lt,
    train_res_enc_lt_auxi,
    train_res_tm,
    train_res_tm_auxi,
)
from .commands.embed import EmbeddingArguments
from .commands.evaluate import EvaluateArguments
from .commands.extract_3di import Extract3DiArguments
from .commands.predict_bipartite import BipartitePredictionArguments
from .commands.predict_block import BlockedPredictionArguments
from .commands.predict_serial import PredictionArguments
from .commands.train import TrainArguments
from .commands.train_res_bidirect import TrainArguments as BidirectTrainArguments
from .commands.train_res_enc_auxi import TrainArguments as AuxiTrainArguments
from .commands.train_res_enc_fs_auxi import TrainArguments as FSAuxiTrainArguments
from .commands.train_res_enc_lt import TrainArguments as LTTrainArguments
from .commands.train_res_enc_lt_auxi import TrainArguments as LTAuxiTrainArguments
from .commands.train_res_tm import TrainArguments as TMTrainArguments
from .commands.train_res_tm_auxi import TrainArguments as TMAuxiTrainArguments

DScriptArguments = (
    EmbeddingArguments
    | EvaluateArguments
    | PredictionArguments
    | BlockedPredictionArguments
    | BipartitePredictionArguments
    | TrainArguments
    | BidirectTrainArguments
    | AuxiTrainArguments
    | FSAuxiTrainArguments
    | LTTrainArguments
    | LTAuxiTrainArguments
    | TMTrainArguments
    | TMAuxiTrainArguments
    | Extract3DiArguments
)


class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __citation__

        print(__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)


def main():
    from . import __version__

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-v", "--version", action="version", version="D-SCRIPT " + __version__
    )
    parser.add_argument(
        "-c",
        "--citation",
        action=CitationAction,
        nargs=0,
        help="show program's citation and exit",
    )

    subparsers = parser.add_subparsers(title="D-SCRIPT Commands", dest="cmd")
    subparsers.required = True

    modules = {
        "train": train,
        "train_res_bidirect": train_res_bidirect,
        "train_res_enc_auxi": train_res_enc_auxi,
        "train_res_enc_fs_auxi": train_res_enc_fs_auxi,
        "train_res_enc_lt": train_res_enc_lt,
        "train_res_enc_lt_auxi": train_res_enc_lt_auxi,
        "train_res_tm": train_res_tm,
        "train_res_tm_auxi": train_res_tm_auxi,
        "embed": embed,
        "evaluate": evaluate,
        "predict_serial": predict_serial,
        "predict": predict_block,
        "predict_bipartite": predict_bipartite,
        "extract-3di": extract_3di,
    }

    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args: DScriptArguments = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
