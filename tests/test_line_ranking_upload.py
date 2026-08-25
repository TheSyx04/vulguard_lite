import unittest
from pathlib import Path
import tempfile

from vulguard_lite.ground_truth_pipeline import _line_ranking_hf_output_path
from vulguard_lite.scripts.upload_line_ranking_results import (
    REQUIRED_RESULTS,
    completed_seed_directories,
)


class LineRankingUploadPathTests(unittest.TestCase):
    def test_derives_config_and_seed_from_checkpoint_path(self):
        path = _line_ranking_hf_output_path(
            repo_name="linux",
            model="simcom",
            checkpoint_path="model_config/linux/simcom/linux_2_3/seed_5",
        )
        self.assertEqual(path, "line_ranking/linux/simcom/linux_2_3/seed_5")

    def test_custom_output_path_takes_precedence(self):
        path = _line_ranking_hf_output_path(
            repo_name="linux",
            model="simcom",
            checkpoint_path=None,
            custom_output="/custom/ranking/path/",
        )
        self.assertEqual(path, "custom/ranking/path")

    def test_accepts_checkpoint_file_path(self):
        path = _line_ranking_hf_output_path(
            repo_name="linux",
            model="simcom",
            checkpoint_path="model_config/linux/simcom/linux_2_3/seed_5/com.pth",
        )
        self.assertEqual(path, "line_ranking/linux/simcom/linux_2_3/seed_5")

    def test_requires_custom_path_for_local_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "hf_output_folder"):
            _line_ranking_hf_output_path(
                repo_name="linux",
                model="simcom",
                checkpoint_path=None,
            )


class ExistingLineRankingResultsTests(unittest.TestCase):
    def test_detects_complete_and_incomplete_seed_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_root = Path(temporary_directory)
            complete = model_root / "linux_0_1" / "seed_1"
            incomplete = model_root / "linux_0_1" / "seed_2"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            for filename in REQUIRED_RESULTS:
                (complete / filename).touch()
            (incomplete / "ranked_ground_truth_summary.json").touch()

            directories, incomplete_results = completed_seed_directories(model_root)

            self.assertEqual(directories, [complete, incomplete])
            self.assertEqual(incomplete_results[0][0], incomplete)
            self.assertEqual(
                set(incomplete_results[0][1]),
                REQUIRED_RESULTS - {"ranked_ground_truth_summary.json"},
            )


if __name__ == "__main__":
    unittest.main()
