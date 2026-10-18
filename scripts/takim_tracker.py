import numpy as np
from collections import Counter, deque
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils import matching
from ultralytics.trackers.basetrack import TrackState
from takim_renk import BILINMIYOR

# ByteTrack + takim kisiti. Ust uste binen iki rakip oyuncuda kutular ortusur, IoU
# hangisinin hangisi oldugunu bilemez ve kimlik rakibe gecer (dev'de 3 dk'da ~100 kez).
# Forma rengi bunu bilir: kirmizi izi lacivert tespitle eslestirmek imkansiz.
# Tespitin takimi ByteTrack'in kullanmadigi `cls` alaninda tasinir (0/1/BILINMIYOR).
# Izin takimi son OY_SAYISI guclu etiketin cogunlugu — tek karelik yanlis okuma
# eslesmeyi engellemesin diye son etiket degil.
OY_SAYISI = 30
YASAK = 1.0          # iou mesafesi 1 = hic ortusme yok; tum esiklerin ustunde


class TakimTracker(BYTETracker):
    def __init__(self, args, kisit=True):
        super().__init__(args)
        self.kisit = kisit                            # False: karsilastirma tabani (ayni yol, kisitsiz)
        self.oylar = {}                               # track_id -> deque

    def _iz_takimi(self, t):
        oy = self.oylar.get(t.track_id)
        if not oy:
            return BILINMIYOR
        takim, n = Counter(oy).most_common(1)[0]
        return takim if n >= 3 else BILINMIYOR       # 3 oydan az: henuz emin degiliz

    def _kisitla(self, dists, tracks, detections):
        if dists.size == 0 or not self.kisit:
            return dists
        tt = np.array([self._iz_takimi(t) for t in tracks])
        dt = np.array([int(d.cls) for d in detections])
        celiski = ((tt[:, None] != BILINMIYOR) & (dt[None, :] != BILINMIYOR)
                   & (tt[:, None] != dt[None, :]))
        dists = dists.copy()
        dists[celiski] = YASAK
        self.n_engel += int(celiski.sum())
        return dists

    def get_dists(self, tracks, detections):
        return self._kisitla(super().get_dists(tracks, detections), tracks, detections)

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
        # ust sinifla ayni, sadece mesafeye takim kisiti eklendi
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked and detections_second:
            dists = self._kisitla(matching.iou_distance(r_tracked, detections_second),
                                  r_tracked, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked, detections_second, activated, refind)
        else:
            u_track = list(range(len(r_tracked)))
        for it in u_track:
            r_tracked[it].mark_lost()
            lost.append(r_tracked[it])

    def update(self, results, img=None, **kw):
        self.n_engel = 0
        out = super().update(results, img, **kw)
        for t in self.tracked_stracks:                # bu karede eslesen izlerin oyu
            if t.frame_id == self.frame_id and int(t.cls) != BILINMIYOR:
                self.oylar.setdefault(t.track_id, deque(maxlen=OY_SAYISI)).append(int(t.cls))
        return out
