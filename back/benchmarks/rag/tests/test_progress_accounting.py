"""Progress honesty tests: parse_progress_line + format_progress_marker on shipped APIs.

These cases complement test_subprocess_runner.py (mocked Popen / Progress) by focusing
on pure parser loops, round-trips, clamp without success reconcile, and noisy input.
"""

from unittest.mock import MagicMock, patch

from core.subprocess_runner import (
    format_progress_marker,
    parse_progress_line,
    run_command_with_progress,
)


def _apply_progress_event(completed: int, total: int, parsed) -> int:
    """Mirror run_command_with_progress clamp semantics for pure-loop tests."""
    if parsed is None:
        return completed
    action, value = parsed
    if action == "set":
        return min(max(value, 0), total)
    if action == "advance":
        return min(completed + value, total)
    return completed


def test_format_progress_marker_roundtrip_parseable_as_generation_step_3():
    """format_progress_marker('generation', 3, 10) must parse as generation set completed=3."""
    marker = format_progress_marker("generation", 3, 10)
    assert parse_progress_line(marker, "generation") == ("set", 3)
    assert parse_progress_line(marker, "retrieval") is None


def test_format_progress_marker_roundtrip_other_stages():
    for stage, completed, total in (
        ("retrieval", 0, 5),
        ("evaluation", 4, 4),
        ("GENERATION", 1, 2),  # stage matching is case-insensitive
    ):
        marker = format_progress_marker(stage, completed, total)
        # step_pattern compared case-insensitively against stage token
        pattern = stage.lower() if stage != "GENERATION" else "generation"
        assert parse_progress_line(marker, pattern) == ("set", completed)


def test_parse_progress_line_set_semantics_and_stage_filter():
    assert parse_progress_line("PROGRESS generation 0/10", "generation") == ("set", 0)
    assert parse_progress_line("PROGRESS generation 10/10", "generation") == ("set", 10)
    # embedded marker still matches via search
    assert parse_progress_line("info: PROGRESS evaluation 2/8 done", "evaluation") == ("set", 2)
    # mismatched stage ignored
    assert parse_progress_line("PROGRESS evaluation 2/8", "generation") is None
    # empty / non-progress
    assert parse_progress_line("", "generation") is None
    assert parse_progress_line("hello world", "generation") is None


def test_n_progress_events_reach_completed_equals_total():
    """N planned steps + N PROGRESS generation events → completed == total (pure loop)."""
    n = 7
    total = n
    completed = 0
    for i in range(1, n + 1):
        line = format_progress_marker("generation", i, total)
        parsed = parse_progress_line(line, "generation")
        completed = _apply_progress_event(completed, total, parsed)
    assert completed == total


def test_n_legacy_generation_advances_reach_total():
    """N Running lines advance generation completed to total without overshoot."""
    n = 5
    completed = 0
    lines = [f"Running B{i}: baseline work" for i in range(n)]
    for line in lines:
        parsed = parse_progress_line(line, "generation")
        assert parsed == ("advance", 1)
        completed = _apply_progress_event(completed, n, parsed)
    assert completed == n


def test_noisy_and_mismatched_lines_do_not_overshoot_total():
    """Clamp: extra advances / wrong-stage markers / noise never push completed > total."""
    total = 3
    completed = 0
    lines = [
        "noise before start",
        format_progress_marker("retrieval", 1, 99),  # wrong stage for generation
        "Running B0: first",
        "Reusing previously generated answer for B1 from checkpoint.",
        format_progress_marker("generation", 99, 3),  # set beyond total → clamp
        "Running B2: extra after clamp",
        "Running command: should not advance",
        "Evaluated case 2/3",  # evaluation pattern, not generation
        "more noise",
        format_progress_marker("generation", 2, 3),
        "Running B3: another advance",
        "Running B4: and another",
    ]
    for line in lines:
        parsed = parse_progress_line(line, "generation")
        completed = _apply_progress_event(completed, total, parsed)
        assert completed <= total
    assert completed == total  # clamped, never overshoots


def test_checkpoint_reuse_line_advances_generation_pattern():
    line = "Reusing previously generated answer for B1 from checkpoint."
    assert parse_progress_line(line, "generation") == ("advance", 1)
    assert parse_progress_line(line, "retrieval") is None
    assert parse_progress_line(line, "evaluation") is None

    # mixed with Running advances in a pure loop
    completed = 0
    total = 3
    for line in (
        "Reusing previously generated answer for B0 from checkpoint.",
        "Running B1: live gen",
        "Reusing previously generated answer for B2 from checkpoint.",
    ):
        completed = _apply_progress_event(
            completed, total, parse_progress_line(line, "generation")
        )
    assert completed == 3


def test_evaluation_set_and_clamp_in_pure_loop():
    total = 4
    completed = 0
    for line in (
        "Evaluated case 1/4",
        "garbage",
        "Evaluated case 3/4",
        "Evaluated case 40/4",  # overshoot set → clamp to total
    ):
        completed = _apply_progress_event(
            completed, total, parse_progress_line(line, "evaluation")
        )
        assert 0 <= completed <= total
    assert completed == total


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_simulated_run_n_progress_markers_ends_completed_eq_total(
    mock_popen, mock_progress_class
):
    """Mocked Popen: N PROGRESS generation markers with total N → completed == N."""
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task

    n = 4
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        f"{format_progress_marker('generation', i, n)}\n" for i in range(1, n + 1)
    ] + [""]
    mock_proc.wait.return_value = 0

    run_command_with_progress(["mock_cmd"], "Gen", n, "generation")

    completed_values = [
        c.kwargs.get("completed")
        for c in mock_progress.update.call_args_list
        if "completed" in c.kwargs
    ]
    assert completed_values
    assert all(v <= n for v in completed_values)
    # last PROGRESS already sets completed to n (also success reconcile keeps n)
    assert completed_values[-1] == n
    assert n in completed_values


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_simulated_run_failure_does_not_force_reconcile_to_total(
    mock_popen, mock_progress_class
):
    """On non-zero exit, incomplete progress is not forced to total."""
    import subprocess
    import pytest

    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task

    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        "PROGRESS generation 1/5\n",
        "Running B1: partial\n",
        "",
    ]
    mock_proc.wait.return_value = 2

    with pytest.raises(subprocess.CalledProcessError):
        run_command_with_progress(["mock_cmd"], "Gen", 5, "generation")

    completed_values = [
        c.kwargs.get("completed")
        for c in mock_progress.update.call_args_list
        if "completed" in c.kwargs
    ]
    assert completed_values
    assert all(v < 5 for v in completed_values)
    assert completed_values[-1] == 2  # set 1 then advance → 2; no success reconcile
