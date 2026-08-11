"""Basic smoke tests for local development.
Run with: python test_app_core.py
"""
import numpy as np
import torch

from config import FALSE_OR_NONBLOCKING_CLASSES
from model import load_phase3_checkpoint
from predict import apply_decision_gate, run_tiled_inference

model, meta = load_phase3_checkpoint('model_bundle/phase3_statewide_deep_learning_model.pt')
assert meta['in_channels'] == 9
assert len(meta['class_names']) == 17
x = torch.zeros((1, 9, 256, 256), dtype=torch.float32, device=next(model.parameters()).device)
with torch.no_grad():
    out = model(x)
assert out['mask_logits'].shape == (1, 1, 256, 256)
assert out['class_logits'].shape[1] == 17
assert out['obstruction_logit'].shape == (1,)

for cls in FALSE_OR_NONBLOCKING_CLASSES:
    gate = apply_decision_gate(0.99, cls, 0.30)
    assert gate['blockade_yn'] == 'No', cls

# Dummy RGB-like stack smoke inference.
stack = np.random.rand(9, 300, 300).astype('float32')
image_meta = {'georeferenced': False, 'transform': None, 'crs': None, 'height': 300, 'width': 300}
df, mask, summary, aux = run_tiled_inference(model, meta, stack, image_meta, tile_size=256, overlap=0.25, max_tiles=2)
assert len(df) == 2
assert mask.shape == (300, 300)
print('All app core smoke tests passed.')
