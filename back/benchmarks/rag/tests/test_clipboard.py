import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.clipboard import copy_to_clipboard, find_runs, get_metrics_summary

def test_copy_to_clipboard_darwin():
    with patch("sys.platform", "darwin"), \
         patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        res = copy_to_clipboard("test text")
        assert res is True
        mock_popen.assert_called_with(["pbcopy"], stdin=subprocess.PIPE)
        mock_proc.communicate.assert_called_with(input=b"test text")

def test_copy_to_clipboard_linux_xclip():
    with patch("sys.platform", "linux"), \
         patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen:
        
        # xclip exists
        mock_run.return_value = MagicMock(returncode=0)
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        
        res = copy_to_clipboard("test text")
        assert res is True
        mock_popen.assert_called_with(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        mock_proc.communicate.assert_called_with(input=b"test text")

def test_copy_to_clipboard_linux_xsel():
    with patch("sys.platform", "linux"), \
         patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen:
        
        # xclip missing, xsel exists
        def side_effect(cmd, *args, **kwargs):
            if "xclip" in cmd:
                return MagicMock(returncode=1)
            elif "xsel" in cmd:
                return MagicMock(returncode=0)
            return MagicMock(returncode=1)
        mock_run.side_effect = side_effect
        
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        
        res = copy_to_clipboard("test text")
        assert res is True
        mock_popen.assert_called_with(["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE)

def test_copy_to_clipboard_windows():
    with patch("sys.platform", "win32"), \
         patch("os.name", "nt"), \
         patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen:
        
        # xclip and xsel missing, nt OS
        mock_run.return_value = MagicMock(returncode=1)
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        
        res = copy_to_clipboard("test text")
        assert res is True
        mock_popen.assert_called_with(["clip"], stdin=subprocess.PIPE)

def test_copy_to_clipboard_exception():
    with patch("sys.platform", "darwin"), \
         patch("subprocess.Popen", side_effect=Exception("error")):
        res = copy_to_clipboard("test text")
        assert res is False

def test_find_runs(tmp_path):
    reports_dir = tmp_path / "reports"
    
    # Missing reports_dir should sys.exit
    with pytest.raises(SystemExit):
        find_runs(reports_dir)
        
    reports_dir.mkdir()
    
    # Create mock dirs
    run1 = reports_dir / "run_20260623_120000"
    run2 = reports_dir / "run_20260623_130000"
    other = reports_dir / "other_dir"
    
    run1.mkdir()
    run2.mkdir()
    other.mkdir()
    
    runs = find_runs(reports_dir)
    assert len(runs) == 2
    # latest first: run2, then run1
    assert runs[0] == run2
    assert runs[1] == run1

def test_get_metrics_summary_exists(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    summary_file = run_dir / "metrics_summary.md"
    summary_file.write_text("existing summary")
    
    res = get_metrics_summary(run_dir, tmp_path, tmp_path)
    assert res == "existing summary"

def test_get_metrics_summary_auto_generate(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    
    result_metrics = run_dir / "result_metrics.yaml"
    result_metrics.write_text("metrics content")
    
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    parse_script = script_dir / "parse_metrics.py"
    parse_script.write_text("# parse script")
    
    summary_file = run_dir / "metrics_summary.md"
    
    def mock_subprocess_run(cmd, **kwargs):
        # simulate script writing the summary file
        summary_file.write_text("generated summary")
        return MagicMock()
        
    with patch("subprocess.run", side_effect=mock_subprocess_run):
        res = get_metrics_summary(run_dir, tmp_path, script_dir)
        assert res == "generated summary"

def test_get_metrics_summary_fallback_global(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    
    # Neither metrics_summary.md nor result_metrics.yaml exists
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    
    global_fallback = reports_dir / "metrics_summary.md"
    global_fallback.write_text("global fallback summary")
    
    res = get_metrics_summary(run_dir, reports_dir, tmp_path)
    assert res == "global fallback summary"

def test_get_metrics_summary_fallback_missing(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    
    # Everything is missing, should exit
    with pytest.raises(SystemExit):
        get_metrics_summary(run_dir, reports_dir, tmp_path)

def test_get_metrics_summary_auto_generate_fail(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    result_metrics = run_dir / "result_metrics.yaml"
    result_metrics.write_text("metrics")
    
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    parse_script = script_dir / "parse_metrics.py"
    parse_script.write_text("# parse script")
    
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    global_fallback = reports_dir / "metrics_summary.md"
    global_fallback.write_text("global fallback summary")

    with patch("subprocess.run", side_effect=Exception("parse error")):
        res = get_metrics_summary(run_dir, reports_dir, script_dir)
        # Should fallback to global
        assert res == "global fallback summary"

def test_get_metrics_summary_no_rich_logging(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    
    # Trigger fallback log without rich
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    global_fallback = reports_dir / "metrics_summary.md"
    global_fallback.write_text("global fallback")
    
    import core.clipboard
    old_rich = core.clipboard.HAS_RICH
    core.clipboard.HAS_RICH = False
    try:
        res = get_metrics_summary(run_dir, reports_dir, tmp_path)
        assert res == "global fallback"
    finally:
        core.clipboard.HAS_RICH = old_rich

def test_get_metrics_summary_auto_generate_no_rich(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    result_metrics = run_dir / "result_metrics.yaml"
    result_metrics.write_text("metrics content")
    
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    parse_script = script_dir / "parse_metrics.py"
    parse_script.write_text("# parse script")
    
    summary_file = run_dir / "metrics_summary.md"
    
    def mock_subprocess_run(cmd, **kwargs):
        summary_file.write_text("generated summary")
        return MagicMock()
        
    import core.clipboard
    old_rich = core.clipboard.HAS_RICH
    core.clipboard.HAS_RICH = False
    try:
        with patch("subprocess.run", side_effect=mock_subprocess_run):
            res = get_metrics_summary(run_dir, tmp_path, script_dir)
            assert res == "generated summary"
    finally:
        core.clipboard.HAS_RICH = old_rich

def test_clipboard_import_no_rich():
    import sys
    import importlib
    with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None, "rich.panel": None}):
        import core.clipboard
        importlib.reload(core.clipboard)
        assert core.clipboard.HAS_RICH is False
        
    # restore
    importlib.reload(core.clipboard)

