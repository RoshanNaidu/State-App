# Web App Design

## Purpose

Provide an upload-based demo tool for applying the Phase 3 trained model to imagery from any Indiana region.

## Core flow

1. User uploads imagery.
2. App builds 9-channel Phase 3 input stack.
3. App tiles the image.
4. App runs the trained Phase 3 model checkpoint.
5. App applies production decision gates.
6. App displays candidate outputs and downloadable files.

## Outputs

- Candidate table CSV
- Overlay image PNG
- Mask heatmap PNG
- GeoJSON for georeferenced GeoTIFF/TIFF input

## Decision gate

False-positive and non-obstruction classes are prevented from becoming final `blockade_yn = Yes`. Model-only `Yes` remains review-required.
