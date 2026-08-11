"""Configuration values derived from the Phase 3 production notebook."""

PHASE3_CHANNEL_NAMES = [
    "red",
    "green",
    "blue",
    "nir_or_zero",
    "ndvi01_or_zero",
    "water_like_proxy",
    "dem01_or_zero",
    "dem_slope01_or_zero",
    "brightness",
]

PHASE3_CLASS_NAMES = [
    "no_obstruction",
    "official_low_head_dam",
    "official_dam_or_control_structure",
    "possible_fallen_tree",
    "possible_log_jam_or_woody_debris",
    "possible_debris_jam_mixed_material",
    "possible_beaver_dam",
    "possible_sediment_gravel_bar",
    "possible_rock_riffle_or_natural_drop",
    "possible_ice_or_seasonal_blockage",
    "bridge_or_road_crossing_false_positive",
    "culvert_or_small_crossing_possible_obstruction",
    "shadow_tree_canopy_false_positive",
    "vegetation_aquatic_growth",
    "dry_or_low_water_channel_artifact",
    "uncertain_obstruction",
    "insufficient_data",
]

TRUE_OBSTRUCTION_CLASSES = {
    "official_low_head_dam",
    "official_dam_or_control_structure",
    "possible_fallen_tree",
    "possible_log_jam_or_woody_debris",
    "possible_debris_jam_mixed_material",
    "possible_beaver_dam",
    "possible_sediment_gravel_bar",
    "possible_ice_or_seasonal_blockage",
    "culvert_or_small_crossing_possible_obstruction",
    "uncertain_obstruction",
}

FALSE_OR_NONBLOCKING_CLASSES = {
    "no_obstruction",
    "possible_rock_riffle_or_natural_drop",
    "bridge_or_road_crossing_false_positive",
    "shadow_tree_canopy_false_positive",
    "vegetation_aquatic_growth",
    "dry_or_low_water_channel_artifact",
    "insufficient_data",
}

REVIEW_ONLY_CLASSES = {
    "uncertain_obstruction",
    "culvert_or_small_crossing_possible_obstruction",
    "possible_beaver_dam",
    "possible_ice_or_seasonal_blockage",
}

DEFAULT_MODEL_PATH = "model_bundle/phase3_statewide_deep_learning_model.pt"
APP_MODEL_VERSION_LABEL = "phase3-production-gated-v1.0 checkpoint"
