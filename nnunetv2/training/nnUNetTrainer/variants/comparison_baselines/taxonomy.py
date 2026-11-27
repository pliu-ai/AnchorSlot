"""Immutable benchmark taxonomy, with no import-time environment configuration."""

LEAF_NAMES = (
    "background", "ecs", "pm", "cyto", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum", "endo_mem", "endo_lum",
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum", "er_mem", "er_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum", "np_out", "np_in",
    "hchrom", "echrom", "nucpl", "mt_out", "mt_in", "perox_mem", "perox_lum",
)
DYNAMIC_GROUP_LABELS = (
    (4, 5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (19, 20), (25, 26), (21, 22), (23, 24), (28, 29), (30, 31), (17, 18),
)
DYNAMIC_GROUP_NAMES = (
    "mito", "golgi", "ves", "endo", "lyso", "ld", "eres", "chromatin",
    "ne", "np", "mt", "perox", "er",
)
SINGLETON_LABELS = (1, 2, 3, 27)
