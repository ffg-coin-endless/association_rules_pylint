"""
Test the `run_pylint_on_repo` function from the `linting` module.
"""
import pandas as pd
from src import linting

def test_run_pylint_on_repo(tmp_path, monkeypatch):
    """
    Test the run_pylint_on_repo function by simulating a repository with a dummy 
    Python file, mocking the list_py_files function to return the dummy file, 
    and monkeypatching subprocess.
    """
    # Fake repo path with dummy .py file
    repo_path = tmp_path
    dummy_file = repo_path / "dummy.py"
    dummy_file.write_text("print('hello')")

    # Monkeypatch list_py_files to return our dummy
    monkeypatch.setattr("src.linting.list_py_files", lambda _: [str(dummy_file)])

    # Monkeypatch subprocess to simulate pylint JSON output
    def fake_run(cmd, stdout, stderr, text):
        stdout.write('[{"path": "dummy.py", "message-id": "E0001", "symbol": "error"}]')
        class Result:
            """
            A simple class representing the result of a process, with a 
            placeholder for standard error output.
            """
            stderr = ""
        return Result()
    monkeypatch.setattr("subprocess.run", fake_run)

    csv_path = linting.run_pylint_on_repo("DummyRepo", str(repo_path))
    assert csv_path and csv_path.endswith(".csv")
    df = pd.read_csv(csv_path)
    assert "message-id" in df.columns
