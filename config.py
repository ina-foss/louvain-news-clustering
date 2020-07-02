PATH = "/usr/src/app/"
DATASET = "event2018_short"
THRESHOLDS = [0.6]
SIMILARITIES = [0.3]
DAYS = [1]
QUALITY_FUNCTION = "SurpriseVertexPartition"
#QUALITY_FUNCTION = "ModularityVertexPartition"
WEIGHTS = [
    {"hashtag": 0, "text": 1, "url": 0},
]
WINDOW_DAYS=1
WRITE_CLUSTERS_TEXT = True
WRITE_CLUSTERS_SMALL_IDS = True