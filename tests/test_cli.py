"""Tests for QLoRA JSON extraction CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from qlora_extraction.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_extract(tmp_path: Path) -> None:
    ocr_file = tmp_path / "sample_ocr.txt"
    ocr_file.write_text("COFFEE SHOP\nDATE: 2026-05-15\nTOTAL $5.50\n", encoding="utf-8")

    result = runner.invoke(app, ["extract", str(ocr_file), "--condition", "qlora"])
    assert result.exit_code == 0
    assert "Extracted Receipt JSON" in result.output
    assert "5.5" in result.output


def test_cli_extract_file_not_found(tmp_path: Path) -> None:
    missing_file = tmp_path / "nonexistent.txt"
    result = runner.invoke(app, ["extract", str(missing_file)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_compare() -> None:
    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 0
    assert "CORD Receipt Structured Extraction Benchmark" in result.output
    assert "Zero-Shot Prompting" in result.output
    assert "4-Bit QLoRA SFT" in result.output


def test_cli_report(tmp_path: Path) -> None:
    out = tmp_path / "qlora_report.md"
    result = runner.invoke(app, ["report", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "QLoRA Fine-Tuning for Structured JSON Extraction Benchmark Report" in content


def test_cli_report_stdout() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert "QLoRA Fine-Tuning for Structured JSON Extraction Benchmark Report" in result.output
