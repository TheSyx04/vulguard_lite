import argparse, os, sys
from .utils.logger import logger
from .utils.utils import SRC_PATH
from .utils.reproducibility import seed_everything
from datetime import datetime
from .training import training
from .evaluating import evaluating
from .experiment import run_experiment
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
    experiment_parser.add_argument("-resume_from_checkpoint", type=str2bool, default=False, help="Resume training from the latest saved checkpoint in -checkpoint_dir (True/False)")
    experiment_parser.add_argument("-checkpoint_dir", type=str, default=None, help="Directory to store epoch checkpoints (default: <dg_cache>/save/<repo_name>/models/checkpoints)")
    experiment_parser.add_argument("-calibrated", type=str2bool, default=True, help="Enable validation threshold calibration in experiment mode (True/False)")
    experiment_parser.add_argument("-threshold", type=float, default=0.5, help="Initial threshold before calibration")
    experiment_parser.add_argument("-budget", type=float_0_1, default=1, help="Marked function budget for calibration in [0, 1]")
    experiment_parser.add_argument("-runs", type=int_gte_1, default=1, help="Number of full train-validate-test experiment runs")
    experiment_parser.add_argument(
        "-calibration_range",
        nargs=3,
        default=None,
        metavar=("START", "END", "STEPS"),
        help="Threshold search range for calibration: START END STEPS (example: -calibration_range 0 1 10001)",
    )
    

    parser = argparse.ArgumentParser(prog="VulGuard", description="A tool for mining, training, evaluating for Just-in-Time Vulnerability Prediction")
    parser.add_argument("-version", action="version", version="%(prog)s " + __version__)
    parser.add_argument("-debug", action="store_true", help="Turn on system debug print")
    parser.add_argument("-log_to_file", action="store_true", help="Logging to file instead of stdout")
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('training', parents=[training_parser], help='Training Function')
    subparsers.add_parser('evaluating', parents=[evaluating_parser], help='Evaluating Function')
    subparsers.add_parser('experiment', parents=[experiment_parser], help='Run full experiment loop: training -> validation calibration -> test')

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
    
    if options.__dict__.get('command') in ['training', 'evaluating', 'inferencing', 'experiment']:
        print(f"Set seed: {options.seed}")
        seed_everything(options.seed)
        
    options.func(options)

if __name__ == "__main__":
    main()
