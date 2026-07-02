"""Tests for evaluation metrics (Phase 11)."""

from pathlib import Path

import numpy as np
import pytest

from utils.evaluation import (
    CloudMetrics,
    EvaluationReport,
    ImageMetrics,
    compute_cloud_metrics,
    compute_mae,
    compute_psnr,
    compute_rmse,
    compute_ssim,
    evaluate,
    save_comparison_image,
    save_csv_report,
    save_json_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_img():
    return np.random.rand(64, 64).astype(np.float32) * 0.8


@pytest.fixture
def rec_img(ref_img):
    return ref_img + np.random.randn(64, 64).astype(np.float32) * 0.05


@pytest.fixture
def cloud_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    return mask


@pytest.fixture
def unresolved_mask(cloud_mask):
    mask = cloud_mask.copy()
    mask[15:25, 15:25] = True  # subset remains unresolved
    return mask


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

class TestPSNR:
    def test_identical(self, ref_img):
        assert compute_psnr(ref_img, ref_img) == float("inf")

    def test_different(self, ref_img, rec_img):
        psnr = compute_psnr(ref_img, rec_img)
        assert psnr > 20.0  # reasonable PSNR

    def test_max_val(self):
        a = np.ones((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        psnr = compute_psnr(a, b, max_val=1.0)
        assert psnr == pytest.approx(0.0, abs=0.1)


class TestSSIM:
    def test_identical(self, ref_img):
        ssim = compute_ssim(ref_img, ref_img)
        assert ssim > 0.99

    def test_range(self, ref_img, rec_img):
        ssim = compute_ssim(ref_img, rec_img)
        assert -1.0 <= ssim <= 1.0

    def test_constant_images(self):
        a = np.ones((32, 32), dtype=np.float32) * 0.5
        b = np.ones((32, 32), dtype=np.float32) * 0.5
        ssim = compute_ssim(a, b)
        assert ssim > 0.9


class TestMAE:
    def test_identical(self, ref_img):
        assert compute_mae(ref_img, ref_img) == pytest.approx(0.0, abs=1e-7)

    def test_known(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        assert compute_mae(a, b) == pytest.approx(1.0)

    def test_symmetric(self, ref_img, rec_img):
        assert compute_mae(ref_img, rec_img) == pytest.approx(compute_mae(rec_img, ref_img))


class TestRMSE:
    def test_identical(self, ref_img):
        assert compute_rmse(ref_img, ref_img) == pytest.approx(0.0, abs=1e-7)

    def test_known(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        assert compute_rmse(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cloud metrics
# ---------------------------------------------------------------------------

class TestCloudMetrics:
    def test_no_clouds(self):
        cm = compute_cloud_metrics(np.zeros((64, 64), dtype=bool))
        assert cm.original_cloud_coverage == 0.0
        assert cm.replacement_percentage == 100.0

    def test_full_unresolved(self, cloud_mask):
        cm = compute_cloud_metrics(cloud_mask, cloud_mask)
        assert cm.unresolved_count == int(cloud_mask.sum())
        assert cm.replacement_percentage == 0.0

    def test_partial(self, cloud_mask, unresolved_mask):
        cm = compute_cloud_metrics(cloud_mask, unresolved_mask)
        assert cm.total_cloudy == int(cloud_mask.sum())
        assert 0.0 < cm.replacement_percentage < 100.0

    def test_to_dict(self, cloud_mask):
        cm = compute_cloud_metrics(cloud_mask)
        d = cm.to_dict()
        assert "original_cloud_coverage" in d
        assert "replacement_percentage" in d


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_basic(self, ref_img, rec_img, cloud_mask, unresolved_mask):
        report = evaluate(ref_img, rec_img, cloud_mask, unresolved_mask, "test.tif")
        assert report.image_name == "test.tif"
        assert report.image_metrics.psnr > 0
        assert report.cloud_metrics is not None

    def test_multiband(self):
        ref = np.random.rand(4, 32, 32).astype(np.float32)
        rec = ref + np.random.randn(4, 32, 32).astype(np.float32) * 0.02
        report = evaluate(ref, rec, image_name="multi.tif")
        assert len(report.per_band_metrics) == 4

    def test_no_cloud_mask(self, ref_img, rec_img):
        report = evaluate(ref_img, rec_img, image_name="nocloud.tif")
        assert report.cloud_metrics is None

    def test_to_dict(self, ref_img, rec_img):
        report = evaluate(ref_img, rec_img, image_name="dict_test.tif")
        d = report.to_dict()
        assert "image_metrics" in d
        assert "per_band" in d


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------

class TestReportExport:
    def test_save_json(self, ref_img, rec_img, tmp_path):
        report = evaluate(ref_img, rec_img, image_name="test")
        out = save_json_report(report, tmp_path / "report.json")
        assert out.exists()
        import json
        with open(out) as f:
            data = json.load(f)
        assert data["image_name"] == "test"

    def test_save_csv(self, ref_img, rec_img, tmp_path):
        reports = [evaluate(ref_img, rec_img, image_name=f"img_{i}") for i in range(3)]
        out = save_csv_report(reports, tmp_path / "reports.csv")
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_save_comparison(self, ref_img, rec_img, tmp_path):
        out = save_comparison_image(ref_img, rec_img, tmp_path / "comp.png")
        assert out.exists()

    def test_empty_csv(self, tmp_path):
        out = save_csv_report([], tmp_path / "empty.csv")
        assert out.exists()
