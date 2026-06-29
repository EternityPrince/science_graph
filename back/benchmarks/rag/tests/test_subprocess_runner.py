import subprocess
import pytest
from unittest.mock import MagicMock, patch
from rich.text import Text
from core.subprocess_runner import IterationSpeedColumn, run_command_with_progress

def test_iteration_speed_column_render_zero_or_none():
    col = IterationSpeedColumn()
    task_mock = MagicMock()
    
    # Test None speed
    task_mock.finished_speed = None
    task_mock.speed = None
    res = col.render(task_mock)
    assert isinstance(res, Text)
    assert res.plain == "- sec/it"
    
    # Test 0 speed
    task_mock.finished_speed = 0
    task_mock.speed = 0
    res = col.render(task_mock)
    assert res.plain == "- sec/it"

def test_iteration_speed_column_render_positive():
    col = IterationSpeedColumn()
    task_mock = MagicMock()
    
    # speed = 2.0 -> 0.50 sec/it
    task_mock.finished_speed = 2.0
    task_mock.speed = 0
    res = col.render(task_mock)
    assert res.plain == "0.50 sec/it"
    
    # speed = 0.5 -> 2.00 sec/it via speed (finished_speed is None)
    task_mock.finished_speed = None
    task_mock.speed = 0.5
    res = col.render(task_mock)
    assert res.plain == "2.00 sec/it"

@patch("subprocess.Popen")
@patch("time.perf_counter", side_effect=[0.0, 1.5, 2.0, 3.0, 4.0, 5.0])
def test_run_command_with_progress_success(mock_perf_counter, mock_popen):
    # Set up mock process
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    
    # Simulate stdout lines
    mock_proc.stdout.readline.side_effect = [
        "Query: 'hello'",
        "Running generation: abc",
        "Evaluated case 3/10",
        "Evaluated case invalid/10", # Triggers exception block in evaluation pattern
        "Some warning printed",
        "Successfully loaded",
        "Ready state reached",
        "evaluation finished",
        ""
    ]
    mock_proc.wait.return_value = 0
    
    # Test retrieval pattern
    elapsed = run_command_with_progress(["mock_cmd"], "Test Progress", 10, "retrieval")
    assert elapsed == 1.5
    
    # Test generation pattern
    mock_proc.stdout.readline.side_effect = [
        "Running test:",
        "Running command: foo", # Should be ignored for progress
        ""
    ]
    run_command_with_progress(["mock_cmd"], "Test Progress", 10, "generation")
    
    # Test evaluation pattern
    mock_proc.stdout.readline.side_effect = [
        "Evaluated case 5/10",
        "Evaluated case invalid/10",
        ""
    ]
    run_command_with_progress(["mock_cmd"], "Test Progress", 10, "evaluation")

@patch("subprocess.Popen")
def test_run_command_with_progress_failure(mock_popen):
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [""]
    mock_proc.wait.return_value = 1  # non-zero return code
    
    with pytest.raises(subprocess.CalledProcessError):
        run_command_with_progress(["mock_cmd"], "Test Progress", 10, "retrieval")
