import subprocess
import pytest
from unittest.mock import MagicMock, patch, call
from rich.text import Text
from core.subprocess_runner import (
    IterationSpeedColumn,
    run_command_with_progress,
    parse_progress_line,
    format_progress_marker,
    PROGRESS_LINE_RE,
)


def test_format_progress_marker():
    assert format_progress_marker("generation", 3, 10) == "PROGRESS generation 3/10"
    assert format_progress_marker("retrieval", 0, 1) == "PROGRESS retrieval 0/1"


def test_parse_progress_line_primary_protocol():
    assert parse_progress_line("PROGRESS generation 3/10", "generation") == ("set", 3)
    assert parse_progress_line("  progress retrieval 7/20  ", "retrieval") == ("set", 7)
    # wrong stage ignored
    assert parse_progress_line("PROGRESS generation 3/10", "retrieval") is None


def test_parse_progress_line_legacy_retrieval():
    assert parse_progress_line("Query: 'hello'", "retrieval") == ("advance", 1)
    assert parse_progress_line("[Q01] Query: 'what?' (B1)", "retrieval") == ("advance", 1)
    assert parse_progress_line("no match here", "retrieval") is None


def test_parse_progress_line_legacy_generation():
    assert parse_progress_line("Running B1: baseline details", "generation") == ("advance", 1)
    assert parse_progress_line("Running command: foo", "generation") is None
    assert parse_progress_line(
        "Reusing previously generated answer for B1 from checkpoint.",
        "generation",
    ) == ("advance", 1)


def test_parse_progress_line_legacy_evaluation():
    assert parse_progress_line("Evaluated case 5/10", "evaluation") == ("set", 5)
    assert parse_progress_line("Evaluated case invalid/10", "evaluation") is None


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
        "Evaluated case invalid/10",  # Triggers exception block in evaluation pattern
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
        "Running command: foo",  # Should be ignored for progress
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


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_generation_reuse_and_progress(mock_popen, mock_progress_class):
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task

    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        "Reusing previously generated answer for B1 from checkpoint.\n",
        "PROGRESS generation 2/10\n",
        "Running B2: something\n",
        ""
    ]
    mock_proc.wait.return_value = 0

    run_command_with_progress(["mock_cmd"], "Title", 10, "generation")

    # reuse advances to 1, PROGRESS set 2, Running advances to 3, success reconcile to 10
    completed_values = [
        c.kwargs.get("completed")
        for c in mock_progress.update.call_args_list
        if "completed" in c.kwargs
    ]
    assert 1 in completed_values
    assert 2 in completed_values
    assert 3 in completed_values
    assert completed_values[-1] == 10  # success reconcile


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_clamp_and_reconcile(mock_popen, mock_progress_class):
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task

    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    # Overshoot via PROGRESS set beyond total, and extra advances
    mock_proc.stdout.readline.side_effect = [
        "PROGRESS retrieval 99/5\n",
        "Query: 'a'\n",
        "Query: 'b'\n",
        ""
    ]
    mock_proc.wait.return_value = 0

    run_command_with_progress(["mock_cmd"], "Title", 5, "retrieval")

    completed_values = [
        c.kwargs.get("completed")
        for c in mock_progress.update.call_args_list
        if "completed" in c.kwargs
    ]
    assert all(v <= 5 for v in completed_values)
    assert completed_values[0] == 5  # clamped set


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_success_reconcile_empty(mock_popen, mock_progress_class):
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task

    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [""]
    mock_proc.wait.return_value = 0

    run_command_with_progress(["mock_cmd"], "Title", 5, "retrieval")

    # No advances but success -> reconcile to total
    mock_progress.update.assert_called_with(mock_task, completed=5)


@patch("subprocess.Popen")
def test_run_command_with_progress_failure(mock_popen):
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [""]
    mock_proc.wait.return_value = 1  # non-zero return code

    with pytest.raises(subprocess.CalledProcessError):
        run_command_with_progress(["mock_cmd"], "Test Progress", 10, "retrieval")
