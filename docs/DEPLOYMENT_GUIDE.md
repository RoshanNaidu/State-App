# Deployment Guide

## Recommended for tomorrow's demo

Use Streamlit Community Cloud:

1. Upload this folder to GitHub.
2. In Streamlit Community Cloud, select the repository.
3. Set the app file to `app.py`.
4. Deploy.

## Why not GitHub Pages?

GitHub Pages cannot run Python/PyTorch inference. It can host a static website only. This project requires server-side Python dependencies such as PyTorch, NumPy, PIL, and optionally rasterio/pyproj for GeoTIFF output.

## Hugging Face Spaces option

This app can also run on Hugging Face Spaces with SDK = Streamlit.

## Production deployment note

A real government deployment should add authentication, audit logs, reviewer workflow, secure storage, model registry/versioning, and an approved infrastructure environment.
