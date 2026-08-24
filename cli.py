import argparse, os, sys
from .utils.logger import logger
from .utils.utils import SRC_PATH
from .utils.reproducibility import seed_everything
from datetime import datetime
from .training import training
from .evaluating import evaluating
from .experiment import run_experiment
from .attribution.runner import attribute
from .attribution.source_provenance import prepare_lines
from .ground_truth_pipeline import prepare_ground_truth_hunks, rank_ground_truth
from .models.init_model import models

__version__ = "0.2.01"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: True/False")


def float_0_1(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Expected a float in range [0, 1]")

    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("Expected a float in range [0, 1]")
    return parsed


def float_gt_0_1(value):
    parsed = float_0_1(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Expected a float in range (0, 1]")
    return parsed


def int_gte_1(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Expected an integer >= 1")

    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected an integer >= 1")
    return parsed


def int_gte_0(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Expected an integer >= 0")

    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected an integer >= 0")
    return parsed


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    available_languages = ["Python", "Java", "C++", "C", "C#", "JavaScript", "TypeScript", "Ruby", "PHP", "Go", "Swift"]
    modes = ["local", "remote"]

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-dg_save_folder", default=".", type=str, help="")
    common_parser.add_argument("-mode", type=str, default="local", help="Mode of extractor", choices=modes)
    common_parser.add_argument("-repo_name", type=str, default=None, help="Repo name")
    common_parser.add_argument("-repo_path", type=str, default=None, help="Path to git repository")
    common_parser.add_argument("-repo_clone_url", type=str, default=None, help="URL to repository")
    common_parser.add_argument("-repo_clone_path", type=str, default=None, help="Path to clone repository")
    common_parser.add_argument("-repo_language", type=str, required=True, default=None, choices=available_languages, help="Main language of repo")
    common_parser.add_argument("-seed", type=int_gte_0, default=42, help="Random seed for reproducible runs")
    common_parser.add_argument("-hf_repo_id", type=str, default=None, help="Hugging Face dataset repo id used when train/val/test files are not passed explicitly")
    common_parser.add_argument("-hf_revision", type=str, default="main", help="Hugging Face dataset revision or branch")
    common_parser.add_argument("-hf_split_path", type=str, default=None, help="Optional subdirectory in the HF dataset repo to pin a specific split/fold")
    common_parser.add_argument("-hf_output_repo_id", type=str, default=None, help="Hugging Face dataset repo id used to upload experiment outputs")
    common_parser.add_argument("-hf_upload_result", type=str2bool, default=False, help="Upload final experiment outputs to Hugging Face dataset repo (True/False)")
    common_parser.add_argument(
        "-hf_upload_checkpoint_only",
        "-hf_upload_model_config_only",
        dest="hf_upload_checkpoint_only",
        type=str2bool,
        default=False,
        help=(
            "Upload only one inference-ready checkpoint per sampling seed (from run 1); "
            "do not upload experiment result files. The old "
            "-hf_upload_model_config_only name is retained as an alias."
        ),
    )
    
    training_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    training_parser.set_defaults(func=training)
    training_parser.add_argument("-model", type=str, default=None, choices=models, help="List of models")
    training_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")
    training_parser.add_argument("-threshold", type=float, default=None, help="Threshold for warning")
    training_parser.add_argument("-epochs",type=int,default=1, help="")
    training_parser.add_argument("-model_path", type=str, default=None, help="Path to pretrain models")
    training_parser.add_argument("-train_set", type=str, default=None, help="")
    training_parser.add_argument("-val_set", type=str, default=None, help="")
    training_parser.add_argument("-hyperparameters",type=str,default=None, help="Path to hyperparameter")
    training_parser.add_argument("-dictionary",type=str,default=None, help="Path to dictionary")
    training_parser.add_argument("-sampling", type=str2bool, default=False, help="Enable random undersampling on the training set (True/False)")
    training_parser.add_argument("-resume_from_checkpoint", type=str2bool, default=False, help="Resume training from the latest saved checkpoint in -checkpoint_dir (True/False)")
    training_parser.add_argument("-checkpoint_dir", type=str, default=None, help="Directory to store epoch checkpoints (default: <dg_cache>/save/<repo_name>/models/checkpoints)")

    evaluating_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    evaluating_parser.set_defaults(func=evaluating)
    evaluating_parser.add_argument("-model", type=str, default=None, choices=models, help="List of models")
    evaluating_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")
    evaluating_parser.add_argument("-threshold", type=float, default=None, help="Threshold for warning")
    evaluating_parser.add_argument("-model_path", type=str, default=None, help="Path to pretrain models")
    evaluating_parser.add_argument("-test_set", type=str, default=None, help="")
    evaluating_parser.add_argument("-size_set", type=str, default=None, help="File include number of added line and deleted line of each commit to get effort metrics.")
    evaluating_parser.add_argument("-hyperparameters",type=str,default=None, help="Path to hyperparameter")
    evaluating_parser.add_argument("-dictionary",type=str,default=None, help="Path to dictionary")
    evaluating_parser.add_argument("-calibrated", type=str2bool, default=False, help="Enable threshold calibration from evaluation scores (True/False)")
    evaluating_parser.add_argument("-budget", type=float_0_1, default=1, help="Marked function budget for calibration in [0, 1]")
    evaluating_parser.add_argument("-runs", type=int_gte_1, default=1, help="Number of evaluation runs to execute")
    evaluating_parser.add_argument(
        "-calibration_range",
        nargs=3,
        default=None,
        metavar=("START", "END", "STEPS"),
        help="Threshold search range for calibration: START END STEPS (example: -calibration_range 0 1 10001)",
    )

    attribution_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    attribution_parser.set_defaults(func=attribute)
    attribution_parser.add_argument("-model", choices=["jitfine", "deepjit", "simcom"], default="jitfine")
    attribution_parser.add_argument("-device", default="cpu", help="Eg: cpu, cuda, cuda:1")
    attribution_parser.add_argument("-threshold", type=float_0_1, default=0.5)
    attribution_parser.add_argument("-model_path", required=True, help="Checkpoint file or directory")
    attribution_parser.add_argument(
        "-test_set",
        required=True,
        help="JITFine/SimCom pair or one DeepJIT merge JSONL file",
    )
    attribution_parser.add_argument("-hyperparameters", required=True)
    attribution_parser.add_argument("-dictionary", default=None, help="Required for DeepJIT and SimCom")
    attribution_parser.add_argument(
        "-line_provenance", default=None,
        help="Optional line_provenance.jsonl for verified changed-line ranking",
    )
    attribution_parser.add_argument(
        "-line_aggregation", choices=["sum", "mean", "max"], default="sum",
    )
    attribution_parser.add_argument(
        "-hunk_chunk_size", "-simcom_chunk_size", dest="hunk_chunk_size",
        type=int_gte_1, default=10,
        help=("Maximum changed source lines per Git-hunk chunk for DeepJIT/SimCom; "
              "-simcom_chunk_size is retained as a deprecated alias"),
    )
    attribution_parser.add_argument("-output_dir", required=True)
    attribution_parser.add_argument(
        "-attention_strategy",
        choices=["last_layer_cls_mean", "all_layers_cls_mean", "attention_rollout"],
        default="last_layer_cls_mean",
    )
    attribution_parser.add_argument("-top_k", type=int_gte_1, default=None)
    attribution_parser.add_argument(
        "-target_class", type=int, choices=[0, 1], default=1,
        help="Class logit targeted by Grad-CAM; ignored by JITFine",
    )
    selection_group = attribution_parser.add_mutually_exclusive_group()
    selection_group.add_argument("-commit_id", default=None)
    selection_group.add_argument("-only_predicted_vulnerable", action="store_true")
    selection_group.add_argument("-all_commits", action="store_true")
    output_group = attribution_parser.add_mutually_exclusive_group()
    output_group.add_argument("-overwrite", action="store_true")
    output_group.add_argument("-resume", action="store_true")

    prepare_lines_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    prepare_lines_parser.set_defaults(func=prepare_lines)
    prepare_selection = prepare_lines_parser.add_mutually_exclusive_group(required=True)
    prepare_selection.add_argument("-commit_id", default=None, help="One commit SHA or GitHub commit URL")
    prepare_selection.add_argument(
        "-commit_urls", default=None,
        help="Text/JSONL file containing commit URLs or SHAs",
    )
    prepare_lines_parser.add_argument("-output_dir", required=True)

    prepare_ground_truth_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    prepare_ground_truth_parser.set_defaults(func=prepare_ground_truth_hunks)
    prepare_ground_truth_parser.add_argument(
        "-ground_truth", required=True, help="XLSX ground-truth workbook",
    )
    prepare_ground_truth_parser.add_argument("-sheet", default="Linux", help="Workbook sheet name")
    prepare_ground_truth_parser.add_argument("-output_dir", required=True)

    rank_ground_truth_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    rank_ground_truth_parser.set_defaults(func=rank_ground_truth)
    rank_ground_truth_parser.add_argument(
        "-prepared_dir", default=None,
        help="Local prepared ground-truth directory (omit when using -hf_ground_truth_path)",
    )
    rank_ground_truth_parser.add_argument(
        "-model", choices=["jitfine", "deepjit", "simcom"], required=True,
    )
    rank_ground_truth_parser.add_argument("-device", default="cpu", help="Eg: cpu, cuda, cuda:1")
    rank_ground_truth_parser.add_argument(
        "-model_path", default=None,
        help="Local checkpoint file/directory (omit when using -hf_checkpoint_path)",
    )
    rank_ground_truth_parser.add_argument(
        "-hf_ground_truth_path", default=None,
        help=("Ground-truth directory inside -hf_repo_id; defaults to "
              "dataset/ground_truth_hunks/<repo_name>"),
    )
    rank_ground_truth_parser.add_argument(
        "-hf_checkpoint_path", default=None,
        help="Checkpoint file/directory inside the Hugging Face dataset repository",
    )
    rank_ground_truth_parser.add_argument(
        "-hf_dictionary_path", default=None,
        help=("CNN dictionary inside the Hugging Face dataset repository; defaults to "
              "dataset/<repo_name>/dict_<repo_name>.jsonl"),
    )
    rank_ground_truth_parser.add_argument(
        "-hf_features_path", default=None,
        help=("JITFine Kamei-feature JSONL inside the Hugging Face dataset repository; "
              "when omitted, the repository test feature file is auto-detected"),
    )
    rank_ground_truth_parser.add_argument("-hyperparameters", required=True)
    rank_ground_truth_parser.add_argument(
        "-dictionary", default=None, help="Required for DeepJIT and SimCom",
    )
    rank_ground_truth_parser.add_argument(
        "-features", default=None,
        help="Local manual feature JSONL required for JITFine local runs",
    )
    rank_ground_truth_parser.add_argument("-output_dir", required=True)
    rank_ground_truth_parser.add_argument("-threshold", type=float_0_1, default=0.5)
    rank_ground_truth_parser.add_argument("-target_class", type=int, choices=[0, 1], default=1)
    rank_ground_truth_parser.add_argument(
        "-line_aggregation", choices=["sum", "mean", "max"], default="sum",
    )
    rank_ground_truth_parser.add_argument(
        "-hunk_chunk_size", type=int_gte_1, default=10,
        help="Maximum changed source lines per attribution chunk",
    )
    rank_ground_truth_parser.add_argument(
        "-attention_strategy",
        choices=["last_layer_cls_mean", "all_layers_cls_mean", "attention_rollout"],
        default="last_layer_cls_mean",
        help="JITFine attention strategy; ignored by CNN models",
    )
    rank_ground_truth_parser.add_argument(
        "-metric_top_k", nargs="+", type=int_gte_1, default=[1, 3, 5, 10],
        help="K values for absolute hit rate, Recall@K, and NDCG@K",
    )
    rank_ground_truth_parser.add_argument(
        "-effort_fraction", type=float_gt_0_1, default=0.2,
        help="LOC/recall fraction for effort-aware metrics (default: 0.2)",
    )
    rank_ground_truth_output = rank_ground_truth_parser.add_mutually_exclusive_group()
    rank_ground_truth_output.add_argument("-overwrite", action="store_true")
    rank_ground_truth_output.add_argument("-resume", action="store_true")

    experiment_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    experiment_parser.set_defaults(func=run_experiment)
    experiment_parser.add_argument("-model", type=str, default=None, choices=models, help="List of models")
    experiment_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")
    experiment_parser.add_argument("-epochs", type=int, default=1, help="")
    experiment_parser.add_argument("-model_path", type=str, default=None, help="Path to pretrain models")
    experiment_parser.add_argument("-train_set", type=str, default=None, help="")
    experiment_parser.add_argument("-val_set", type=str, default=None, help="Validation set path used by training and calibration")
    experiment_parser.add_argument("-test_set", type=str, default=None, help="Final test set path")
    experiment_parser.add_argument("-hyperparameters", type=str, default=None, help="Path to hyperparameter")
    experiment_parser.add_argument("-dictionary", type=str, default=None, help="Path to dictionary")
    experiment_parser.add_argument("-sampling", type=str2bool, default=False, help="Enable random undersampling on the training set (True/False)")
    experiment_parser.add_argument("-sampling_seed", type=int_gte_0, default=None, help="Optional fixed seed for undersampling; defaults to -seed when omitted")
    experiment_parser.add_argument(
        "-sampling_seeds",
        nargs="+",
        type=int_gte_0,
        default=None,
        help=(
            "List of undersampling seeds for multi-seed mode. When provided, the experiment "
            "runs -runs times for EACH seed, so total runs = len(seeds) × runs. "
            "Overrides -sampling_seed when set. Example: -sampling_seeds 1 2 3 4 5"
        ),
    )
    experiment_parser.add_argument("-resume_from_checkpoint", type=str2bool, default=False, help="Resume training from the latest saved checkpoint in -checkpoint_dir (True/False)")
    experiment_parser.add_argument("-checkpoint_dir", type=str, default=None, help="Directory to store epoch checkpoints (default: <dg_cache>/save/<repo_name>/models/checkpoints)")
    experiment_parser.add_argument("-calibrated", type=str2bool, default=True, help="Enable validation threshold calibration in experiment mode (True/False)")
    experiment_parser.add_argument("-threshold", type=float, default=0.5, help="Initial threshold before calibration")
    experiment_parser.add_argument(
        "-budget",
        nargs="+",
        type=float_0_1,
        default=[1],
        help="Marked function budget(s) for calibration in [0, 1]",
    )
    experiment_parser.add_argument("-runs", type=int_gte_1, default=1, help="Number of full train-validate-test experiment runs")
    experiment_parser.add_argument(
        "-calibration_range",
        nargs=3,
        default=None,
        metavar=("START", "END", "STEPS"),
        help="Threshold search range for calibration: START END STEPS (example: -calibration_range 0 1 10001)",
    )
    experiment_parser.add_argument(
        "-hf_output_folder",
        type=str,
        default=None,
        help=(
            "Custom remote folder path inside the HF dataset repo where experiment results are uploaded "
            "(e.g. 'output/linux/deepjit/sampling/deepjit_linux_1_3_sampling'). "
            "When omitted the path is derived automatically from the model, repo, split, and sampling settings."
        ),
    )
    parser = argparse.ArgumentParser(prog="VulGuard", description="A tool for mining, training, evaluating for Just-in-Time Vulnerability Prediction")
    parser.add_argument("-version", action="version", version="%(prog)s " + __version__)
    parser.add_argument("-debug", action="store_true", help="Turn on system debug print")
    parser.add_argument("-log_to_file", action="store_true", help="Logging to file instead of stdout")
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('training', parents=[training_parser], help='Training Function')
    subparsers.add_parser('evaluating', parents=[evaluating_parser], help='Evaluating Function')
    subparsers.add_parser('experiment', parents=[experiment_parser], help='Run full experiment loop: training -> validation calibration -> test')
    subparsers.add_parser(
        'attribute',
        parents=[attribution_parser],
        help='Attribute JITFine tokens or DeepJIT/SimCom code-change rows',
    )
    subparsers.add_parser(
        'prepare-lines', parents=[prepare_lines_parser],
        help='Prepare provenance-rich merge/patch inputs from Git commits',
    )
    subparsers.add_parser(
        'prepare-ground-truth-hunks', parents=[prepare_ground_truth_parser],
        help='Create reusable ground-truth hunks from an XLSX sheet and local Git clone',
    )
    subparsers.add_parser(
        'rank-ground-truth', parents=[rank_ground_truth_parser],
        help='Rank prepared ground-truth hunks with JITFine, DeepJIT, or SimCom',
    )

    options = parser.parse_args(args)

    if not options.debug:
        logger.disable()

    if options.log_to_file:
        # Create a folder named 'logs' if it doesn't exist
        if not os.path.exists(f"{SRC_PATH}/logs"):
            os.makedirs(f"{SRC_PATH}/logs")

        # Define a file to log IceCream output
        log_file_path = os.path.join(f"{SRC_PATH}/logs", "logs.log")

        # Replace logging configuration with IceCream configuration
        logger.configureOutput(
            prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ',
            outputFunction=lambda x: open(log_file_path, "a").write(x + "\n"),
        )

    if not hasattr(options, 'func'):
        parser.print_help()
        exit(1)
    
    if options.__dict__.get('command') in [
        'training', 'evaluating', 'experiment', 'attribute', 'prepare-lines',
        'prepare-ground-truth-hunks', 'rank-ground-truth',
    ]:
        print(f"Set seed: {options.seed}")
        seed_everything(options.seed)
        
    options.func(options)

if __name__ == "__main__":
    main()
