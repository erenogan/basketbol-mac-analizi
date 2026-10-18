import json
from pathlib import Path


def video_yolu(ad):
    """out/<ad>/meta.json'daki video (etiketli deney klasorleri icin), yoksa data/<ad>.mp4."""
    meta = Path("out") / ad / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["video"]
    return f"data/{ad}.mp4"
