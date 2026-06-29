import pytest
import yaml
from pathlib import Path
from core.models import parse_report, load_report_file, ReportOutput

def test_parse_report_dict_without_results():
    data = {
        "metadata": {"version": "1.0"},
        "summary": {"total_cases": 0}
    }
    report = parse_report(data)
    assert isinstance(report, ReportOutput)
    assert report.metadata == {"version": "1.0"}
    assert report.summary == {"total_cases": 0}
    assert len(report.results) == 0

def test_parse_report_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported report format"):
        parse_report("some string")

def test_load_report_file_success(tmp_path):
    report_file = tmp_path / "report.yaml"
    data = {
        "metadata": {"version": "2.0"},
        "results": [
            {
                "id": "case_1",
                "query": "What is gravity?",
                "category": "physics",
                "baselines": {}
            }
        ]
    }
    with open(report_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
        
    report = load_report_file(report_file)
    assert report.metadata == {"version": "2.0"}
    assert len(report.results) == 1
    assert report.results[0].id == "case_1"

def test_load_report_file_missing():
    with pytest.raises(FileNotFoundError):
        load_report_file(Path("non_existent_file.yaml"))
