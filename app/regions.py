"""Functional brain systems used everywhere: Modal region summaries, the viewer's bar chart,
and the Gemini agent's target selection. Single source of truth (see CONTRACTS.md).

Region names are bare atlas labels (no hemisphere prefix); scripts/build_assets.py expands each to
L_<name> and R_<name> when it writes atlas.json. Mappings are deliberately coarse: this is a demo
built on a group-average encoding model, not a localizer study.
"""

SYSTEMS = [
    {
        "id": "early_visual",
        "label": "Early visual cortex (V1/V2)",
        "blurb": "Edges, contrast and fine detail. Lights up for any rich visual scene.",
        "game_target": True,
        "game_hint": "fast high-contrast pattern discrimination: spot the odd Gabor, flicker detection, find the tilted line among distractors",
        "destrieux": ["S_calcarine", "G_cuneus", "Pole_occipital", "S_oc_sup_and_transversal", "G_occipital_sup"],
        "glasser": ["V1", "V2", "V3", "V4"],
    },
    {
        "id": "motion",
        "label": "Visual motion (MT+/V5)",
        "blurb": "Tracks moving things: optic flow, moving objects, camera pans.",
        "game_target": True,
        "game_hint": "multiple-object tracking, dodging moving obstacles, judging the direction of coherent dot motion",
        "destrieux": ["G_occipital_middle", "S_oc_middle_and_Lunatus", "S_temporal_inf"],
        "glasser": ["MT", "MST", "FST", "V4t", "LO1", "LO2", "V3A", "V3B"],
    },
    {
        "id": "faces",
        "label": "Face processing (fusiform / FFA)",
        "blurb": "Recognising faces and expressions. Quiet when nobody is on screen.",
        "game_target": True,
        "game_hint": "face memory: match procedurally drawn cartoon faces, spot the changed expression, remember which face you saw",
        "destrieux": ["G_oc-temp_lat-fusifor"],
        "glasser": ["FFC", "PIT", "VVC"],
    },
    {
        "id": "places",
        "label": "Scenes and navigation (PPA / RSC)",
        "blurb": "Layouts, rooms, landmarks and knowing where you are.",
        "game_target": True,
        "game_hint": "top-down maze navigation, remember a route through rooms, spot which landmark moved",
        "destrieux": ["G_oc-temp_med-Parahip", "G_oc-temp_med-Lingual", "S_oc-temp_med_and_Lingual", "S_parieto_occipital", "G_cingul-Post-ventral"],
        "glasser": ["PHA1", "PHA2", "PHA3", "VMV1", "VMV2", "VMV3", "RSC", "POS1", "ProS"],
    },
    {
        "id": "objects",
        "label": "Object recognition (lateral occipital / IT)",
        "blurb": "Shapes and objects as things, not just edges.",
        "game_target": True,
        "game_hint": "mental rotation: does the rotated shape match, silhouette matching, spot the object that changed",
        "destrieux": ["G_and_S_occipital_inf", "S_oc-temp_lat", "G_temporal_inf"],
        "glasser": ["LO3", "V8", "PH", "TE2p"],
    },
    {
        "id": "attention",
        "label": "Spatial attention (intraparietal / SPL)",
        "blurb": "Where to look next, counting, tracking several things at once.",
        "game_target": True,
        "game_hint": "visual search among distractors, rapid counting, keep track of 3 of 8 identical moving discs",
        "destrieux": ["S_intrapariet_and_P_trans", "G_parietal_sup", "G_pariet_inf-Supramar"],
        "glasser": ["LIPv", "LIPd", "VIP", "AIP", "MIP", "IP1", "IP2", "7PC", "7AL", "7Am", "7PL", "PFt"],
    },
    {
        "id": "auditory",
        "label": "Auditory cortex (Heschl's / STG)",
        "blurb": "Sound: pitch, rhythm, voices, music.",
        "game_target": True,
        "game_hint": "pitch ordering, tap along to a beat generated with WebAudio, which tone was different",
        "destrieux": ["G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral", "S_temporal_transverse"],
        "glasser": ["A1", "MBelt", "LBelt", "PBelt", "RI", "A4", "A5", "TA2"],
    },
    {
        "id": "somatomotor",
        "label": "Sensorimotor cortex",
        "blurb": "Planning and feeling movement, hands and body.",
        "game_target": True,
        "game_hint": "rhythm tapping with alternating keys, rapid reaction taps, mirror the hand pose sequence",
        "destrieux": ["G_precentral", "G_postcentral", "S_central", "G_and_S_paracentral"],
        "glasser": ["4", "3a", "3b", "1", "2", "6d", "6v", "6mp", "6ma"],
    },
    {
        "id": "frontal_control",
        "label": "Executive control (dorsolateral prefrontal)",
        "blurb": "Working memory, rules, holding a plan in mind.",
        "game_target": True,
        "game_hint": "n-back with shapes or positions, rule-switching sorting task, remember and reproduce a growing sequence",
        "destrieux": ["G_front_middle", "S_front_inf", "S_front_sup"],
        "glasser": ["p9-46v", "46", "a9-46v", "9-46d", "8C", "8Av", "i6-8", "IFSa", "a47r", "p47r", "8BL", "9a", "9p"],
    },
    {
        "id": "language",
        "label": "Language network (IFG / STS)",
        "blurb": "Speech and meaning. In this build the text pathway is off, so treat as narration only.",
        "game_target": False,
        "game_hint": "",
        "destrieux": ["G_front_inf-Opercular", "G_front_inf-Triangul", "S_temporal_sup", "G_temporal_middle"],
        "glasser": ["44", "45", "55b", "STSdp", "STSda", "STSvp", "STSva", "STGa", "IFJa", "IFSp", "TGd", "TE1a"],
    },
    {
        "id": "default_mode",
        "label": "Default mode network",
        "blurb": "Mind-wandering, self-reference, thinking about others. Tends to switch off when the world is loud.",
        "game_target": False,
        "game_hint": "",
        "destrieux": ["G_cingul-Post-dorsal", "G_precuneus", "G_pariet_inf-Angular", "G_and_S_cingul-Ant", "G_rectus", "G_subcallosal", "S_subparietal"],
        "glasser": ["PCV", "31pv", "31pd", "31a", "v23ab", "d23ab", "23d", "7m", "POS2", "PGi", "PGs", "a24", "p24", "d32", "p32", "10r", "10v", "9m", "8BM"],
    },
]

SYSTEM_IDS = [s["id"] for s in SYSTEMS]
GAME_TARGETS = [s["id"] for s in SYSTEMS if s["game_target"]]


def by_id(system_id: str) -> dict:
    for s in SYSTEMS:
        if s["id"] == system_id:
            return s
    raise KeyError(system_id)
