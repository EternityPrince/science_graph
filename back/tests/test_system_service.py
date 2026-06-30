import tempfile
from pathlib import Path
from unittest.mock import patch

from src.services.system_service import reset_system


def test_reset_system_success():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_file = temp_path / "test.db"
        db_file.touch()

        # Side files
        wal_file = temp_path / "test.db-wal"
        wal_file.touch()
        shm_file = temp_path / "test.db-shm"
        shm_file.touch()

        # USearch file
        usearch_file = temp_path / "test.usearch"
        usearch_file.touch()

        # Archive directory with files and subdirectories
        archive_dir = temp_path / "archive"
        archive_dir.mkdir()
        
        child_file = archive_dir / "doc.pdf"
        child_file.touch()
        
        child_subdir = archive_dir / "subdir"
        child_subdir.mkdir()
        sub_child_file = child_subdir / "sub_doc.pdf"
        sub_child_file.touch()

        # Run reset_system
        report = reset_system(str(db_file), str(archive_dir))

        # Check that everything is deleted
        assert not db_file.exists()
        assert not wal_file.exists()
        assert not shm_file.exists()
        assert not usearch_file.exists()
        # The archive directory itself remains, but it should be empty
        assert archive_dir.exists()
        assert len(list(archive_dir.iterdir())) == 0

        # Verify report
        ops = {r["operation"]: r for r in report}
        assert ops["Delete SQLite Database"]["success"] is True
        assert ops["Delete SQLite -WAL"]["success"] is True
        assert ops["Delete SQLite -SHM"]["success"] is True
        assert ops["Delete USearch Index"]["success"] is True
        assert ops["Clear Archive Directory"]["success"] is True


def test_reset_system_nonexistent():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_file = temp_path / "nonexistent.db"
        archive_dir = temp_path / "nonexistent_archive"

        # Run reset_system - should not raise error and report should be empty
        report = reset_system(str(db_file), str(archive_dir))
        assert len(report) == 0


def test_reset_system_errors():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_file = temp_path / "test.db"
        db_file.touch()
        
        archive_dir = temp_path / "archive"
        archive_dir.mkdir()
        child_file = archive_dir / "doc.pdf"
        child_file.touch()

        (temp_path / "test.db-wal").touch()
        (temp_path / "test.db-shm").touch()
        (temp_path / "test.usearch").touch()

        # Mock Path.unlink to raise OSError
        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            report = reset_system(str(db_file), str(archive_dir))
            
            ops = {r["operation"]: r for r in report}
            assert ops["Delete SQLite Database"]["success"] is False
            assert "Permission denied" in ops["Delete SQLite Database"]["detail"]
            assert ops["Delete SQLite -WAL"]["success"] is False
            assert "Permission denied" in ops["Delete SQLite -WAL"]["detail"]
            assert ops["Delete SQLite -SHM"]["success"] is False
            assert "Permission denied" in ops["Delete SQLite -SHM"]["detail"]
            assert ops["Delete USearch Index"]["success"] is False
            assert "Permission denied" in ops["Delete USearch Index"]["detail"]

        # Mock shutil.rmtree to raise PermissionError
        with patch("shutil.rmtree", side_effect=PermissionError("Failed")):
            # Create a directory to trigger rmtree
            child_dir = archive_dir / "subdir"
            child_dir.mkdir()
            
            report = reset_system(str(db_file), str(archive_dir))
            
            ops = {r["operation"]: r for r in report}
            assert ops["Clear Archive Directory"]["success"] is False
            assert "Failed" in ops["Clear Archive Directory"]["detail"]
