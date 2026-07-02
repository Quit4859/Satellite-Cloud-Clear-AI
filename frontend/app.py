"""Streamlit frontend for Satellite Cloud Clear AI.

Run with::

    streamlit run frontend/app.py
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Satellite Cloud Clear AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🛰️ Satellite Cloud Clear AI")
    st.markdown("---")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    model_name = st.selectbox(
        "AI Model",
        ["placeholder", "prithvi", "satmae", "diffusion", "controlnet"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Settings**")
    compute_metrics = st.checkbox("Compute metrics", value=True)
    save_viz = st.checkbox("Generate visualizations", value=True)
    st.markdown("---")
    st.caption("v1.0.0 — Production Ready")


# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------

st.title("Satellite Cloud Clear AI")
st.markdown("Upload cloudy satellite imagery and get cloud-free reconstructions.")

tab_upload, tab_results, tab_about = st.tabs(["Upload & Process", "Results", "About"])

# ------------------------------------------------------------------
# Tab: Upload & Process
# ------------------------------------------------------------------

with tab_upload:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        uploaded = st.file_uploader(
            "Upload cloudy GeoTIFF",
            type=["tif", "tiff"],
            key="cloudy",
        )
        prev_file = st.file_uploader(
            "Previous temporal image (optional)",
            type=["tif", "tiff"],
            key="prev",
        )
        next_file = st.file_uploader(
            "Next temporal image (optional)",
            type=["tif", "tiff"],
            key="next",
        )
        ref_file = st.file_uploader(
            "Reference image for evaluation (optional)",
            type=["tif", "tiff"],
            key="ref",
        )

    with col2:
        st.subheader("Preview")
        if uploaded is not None:
            try:
                import rasterio

                with rasterio.open(io.BytesIO(uploaded.read())) as src:
                    data = src.read()
                    uploaded.seek(0)
                    if data.shape[0] >= 3:
                        rgb = np.stack([data[2], data[1], data[0]], axis=-1)
                    else:
                        rgb = np.stack([data[0], data[0], data[0]], axis=-1) if data.ndim == 3 else data
                    vmin, vmax = float(np.percentile(rgb, 2)), float(np.percentile(rgb, 98))
                    rgb_norm = np.clip((rgb - vmin) / (vmax - vmin + 1e-10), 0, 1)
                    st.image(rgb_norm, caption=f"Uploaded: {uploaded.name}", use_container_width=True)
                    st.caption(f"Bands: {data.shape[0]}, Size: {data.shape[1]}x{data.shape[2]}")
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")
        else:
            st.info("Upload a GeoTIFF to see preview")

    st.markdown("---")

    if uploaded is not None:
        if st.button("🚀 Run Reconstruction", type="primary", use_container_width=True):
            with st.spinner("Uploading files..."):
                try:
                    import requests

                    files_to_upload = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = requests.post(f"{api_url}/api/upload", files=files_to_upload, timeout=60)
                    resp.raise_for_status()
                    file_id = resp.json()["file_id"]
                    st.success(f"Uploaded: {file_id}")

                    prev_id = None
                    if prev_file is not None:
                        r = requests.post(f"{api_url}/api/upload", files={"file": (prev_file.name, prev_file.getvalue())}, timeout=60)
                        r.raise_for_status()
                        prev_id = r.json()["file_id"]

                    next_id = None
                    if next_file is not None:
                        r = requests.post(f"{api_url}/api/upload", files={"file": (next_file.name, next_file.getvalue())}, timeout=60)
                        r.raise_for_status()
                        next_id = r.json()["file_id"]

                    ref_id = None
                    if ref_file is not None:
                        r = requests.post(f"{api_url}/api/upload", files={"file": (ref_file.name, ref_file.getvalue())}, timeout=60)
                        r.raise_for_status()
                        ref_id = r.json()["file_id"]

                except requests.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running?")
                    st.stop()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.stop()

            with st.spinner("Processing..."):
                try:
                    payload = {
                        "file_id": file_id,
                        "model_name": model_name,
                        "previous_file_id": prev_id,
                        "next_file_id": next_id,
                        "reference_file_id": ref_id,
                        "compute_metrics": compute_metrics,
                        "save_visualizations": save_viz,
                    }
                    resp = requests.post(f"{api_url}/api/process", json=payload, timeout=30)
                    resp.raise_for_status()
                    job_id = resp.json()["job_id"]
                    st.session_state["current_job"] = job_id
                    st.success(f"Job started: {job_id}")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i in range(120):
                        time.sleep(2)
                        r = requests.get(f"{api_url}/api/status/{job_id}", timeout=10)
                        status = r.json()
                        progress_bar.progress(min((i + 1) / 60, 1.0))
                        status_text.text(f"Status: {status['status']}")

                        if status["status"] == "completed":
                            st.success("Processing complete!")
                            st.session_state["job_result"] = status
                            progress_bar.progress(1.0)
                            break
                        elif status["status"] == "failed":
                            st.error(f"Failed: {status.get('message', 'Unknown error')}")
                            break
                    else:
                        st.warning("Processing is taking longer than expected. Check Results tab.")

                except Exception as e:
                    st.error(f"Processing failed: {e}")


# ------------------------------------------------------------------
# Tab: Results
# ------------------------------------------------------------------

with tab_results:
    job_result = st.session_state.get("job_result")
    job_id = st.session_state.get("current_job")

    if job_result is None:
        st.info("No results yet. Run a reconstruction first.")
    else:
        st.subheader(f"Job: {job_id}")
        st.json(job_result)

        if job_result.get("metrics"):
            st.subheader("Evaluation Metrics")
            m = job_result["metrics"]
            cols = st.columns(4)
            cols[0].metric("PSNR", f"{m['psnr']:.2f} dB")
            cols[1].metric("SSIM", f"{m['ssim']:.4f}")
            cols[2].metric("MAE", f"{m['mae']:.6f}")
            cols[3].metric("RMSE", f"{m['rmse']:.6f}")

            cols2 = st.columns(3)
            cols2[0].metric("Cloud Coverage", f"{m['cloud_coverage']:.2%}")
            cols2[1].metric("Replacement %", f"{m['replacement_percentage']:.2f}%")
            cols2[2].metric("Unresolved %", f"{m['unresolved_percentage']:.2f}%")

        if job_result.get("output_files"):
            st.subheader("Download Outputs")
            for label, path in job_result["output_files"].items():
                filename = Path(path).name
                url = f"{api_url}/api/download/{job_id}/{filename}"
                st.markdown(f"- [{label}]({url})")


# ------------------------------------------------------------------
# Tab: About
# ------------------------------------------------------------------

with tab_about:
    st.subheader("About")
    st.markdown("""
    **Satellite Cloud Clear AI** is a production-ready system for removing clouds
    from satellite imagery using temporal reconstruction and AI foundation models.

    ### Pipeline Stages
    1. **Load** — Read GeoTIFF input
    2. **Preprocess** — Normalize and fill NaN values
    3. **Cloud Detection** — Otsu thresholding with morphological filtering
    4. **Registration** — ECC/ORB alignment of temporal images
    5. **Temporal Reconstruction** — Replace cloudy pixels from adjacent dates
    6. **AI Refinement** — Foundation model enhancement (Prithvi, SatMAE, etc.)
    7. **Evaluation** — PSNR, SSIM, MAE, RMSE metrics
    8. **Visualization** — RGB, false color, before/after, difference maps

    ### Supported Models
    - **Placeholder** — Pass-through for testing
    - **Prithvi** — NASA/IBM geospatial foundation model
    - **SatMAE** — Satellite Masked Autoencoder
    - **Diffusion** — Denoising diffusion model
    - **ControlNet** — Mask-conditioned inpainting

    ### Architecture
    - Backend: FastAPI with Pydantic schemas
    - Frontend: Streamlit dashboard
    - Models: Abstract base class with factory pattern
    - Pipeline: Configurable stage-based inference
    """)
