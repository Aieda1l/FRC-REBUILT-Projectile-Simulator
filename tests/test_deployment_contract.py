import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_vercel_runtime_requirements_exclude_offline_calibration_stack(self):
        runtime = (ROOT / "requirements.txt").read_text().splitlines()
        normalized = {line.strip().lower() for line in runtime if line.strip() and not line.startswith("#")}
        self.assertNotIn("scipy", normalized)

        calibration = (ROOT / "requirements-calibration.txt").read_text().splitlines()
        calibration_normalized = {line.strip().lower() for line in calibration if line.strip() and not line.startswith("#")}
        self.assertIn("-r requirements.txt", calibration_normalized)
        self.assertIn("scipy", calibration_normalized)

    def test_offline_calibration_module_is_not_a_vercel_api_entrypoint(self):
        self.assertFalse((ROOT / "api" / "calibration.py").exists())
        self.assertTrue((ROOT / "calibration" / "fitting.py").exists())

    def test_ci_installs_offline_calibration_requirements(self):
        workflow = (ROOT / ".github" / "workflows" / "physics-regression.yml").read_text()
        self.assertIn("pip install -r requirements-calibration.txt", workflow)

    def test_vercel_excludes_offline_and_test_files_from_python_functions(self):
        config = json.loads((ROOT / "vercel.json").read_text())
        functions = config.get("functions", {})
        python = functions.get("api/**/*.py", {})
        excluded = python.get("excludeFiles", "")
        for path in ("tests/**", "docs/**", "scripts/**", "calibration/**"):
            self.assertIn(path, excluded)


if __name__ == "__main__":
    unittest.main()
