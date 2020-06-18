PATH = ""
DATASET = "event2018_short"
THRESHOLDS = [0.6, 0.7]
SIMILARITIES = [0.3, 0.4, 0.5, 0.6, 0.7]
DAYS = [1, 2, 3, 4, 5]
QUALITY_FUNCTION = "SurpriseVertexPartition"
WEIGHTS = [
    {"hashtag": 0, "text": 1, "url": 0},
    {"hashtag": 1, "text": 0, "url": 0},
    {"hashtag": 0, "text": 0, "url": 1},
    {"hashtag": 0, "text": 0.9, "url": 0.1},
    {"hashtag": 0, "text": 0.8, "url": 0.2},
    {"hashtag": 0.1, "text": 0.8, "url": 0.1},
    {"hashtag": 0.2, "text": 0.8, "url": 0},
    {"hashtag": 0, "text": 0.7, "url": 0.3},
    {"hashtag": 0.1, "text": 0.7, "url": 0.2},
    {"hashtag": 0, "text": 0.6, "url": 0.4},
    {"hashtag": 0.1, "text": 0.6, "url": 0.3},
    {"hashtag": 0.2, "text": 0.6, "url": 0.2}
]
WINDOW_DAYS=1
WRITE_CLUSTERS_TEXT = False
WRITE_CLUSTERS_SMALL_IDS = False