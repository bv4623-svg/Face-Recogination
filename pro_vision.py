import cv2, numpy as np, os, time, sqlite3, csv
from datetime import datetime, date
from ultralytics import YOLO
from insightface.app import FaceAnalysis

try:
    import mediapipe as mp
    HAS_MP = True
except Exception as e:
    mp = None; HAS_MP = False
    print("[WARN] mediapipe not available:", e)

# ================= CONFIG =================
KNOWN_DIR      = "known_faces"
SNAPSHOT_DIR   = "snapshots"
DB_PATH        = "face_log.db"
CAMERA_ID      = 0

# --- YOLO settings ---
YOLO_MODEL     = "yolo11s.pt"   # n=fastest, s=balanced, m=accurate
IMG_SIZE       = 640            # 640 standard, 800/1024 slower+accuracy
YOLO_CONF      = 0.25
YOLO_IOU       = 0.45

# --- Face settings ---
FACE_THRESHOLD = 0.42
LOG_COOLDOWN   = 10.0
UNKNOWN_ALERT  = 30.0

EMOTION_MODEL  = "emotion-ferplus-8.onnx"
EMOTION_LABELS = ["neutral","happiness","surprise","sadness",
                  "anger","disgust","fear","contempt"]


class DB:
    def __init__(self, path):
        self.c = sqlite3.connect(path, check_same_thread=False)
        cur = self.c.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            object_type TEXT, name TEXT, emotion TEXT,
            confidence REAL, similarity REAL, track_id INTEGER,
            is_known INTEGER)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT,
            name TEXT, first_seen TEXT, UNIQUE(date,name))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS unknown_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, snapshot TEXT)""")
        self.c.commit()

    def log_person(self, name, emo, conf, sim, known):
        ts = datetime.now().isoformat(timespec="seconds")
        self.c.execute("INSERT INTO detections(timestamp,object_type,name,emotion,confidence,similarity,track_id,is_known) VALUES (?,?,?,?,?,?,?,?)",
                       (ts, "person", name, emo, float(conf), float(sim), -1, 1 if known else 0))
        if known:
            try:
                self.c.execute("INSERT INTO attendance(date,name,first_seen) VALUES (?,?,?)",
                               (date.today().isoformat(), name, ts))
            except sqlite3.IntegrityError:
                pass
        self.c.commit()

    def log_object(self, obj, conf):
        ts = datetime.now().isoformat(timespec="seconds")
        self.c.execute("INSERT INTO detections(timestamp,object_type,name,emotion,confidence,similarity,track_id,is_known) VALUES (?,?,?,?,?,?,?,?)",
                       (ts, obj, "", "", float(conf), 0.0, -1, 0))
        self.c.commit()

    def log_unknown(self, snap):
        self.c.execute("INSERT INTO unknown_alerts(timestamp,snapshot) VALUES (?,?)",
                       (datetime.now().isoformat(timespec="seconds"), snap))
        self.c.commit()

    def stats(self):
        d = date.today().isoformat()
        cur = self.c.cursor()
        cur.execute("SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (d+"%",))
        det = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (d,))
        att = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM unknown_alerts WHERE timestamp LIKE ?", (d+"%",))
        unk = cur.fetchone()[0]
        return det, att, unk

    def export_csv(self):
        path = "attendance_" + date.today().isoformat() + ".csv"
        cur = self.c.cursor()
        cur.execute("SELECT date,name,first_seen FROM attendance ORDER BY date DESC")
        rows = cur.fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["Date","Name","First Seen"]); w.writerows(rows)
        return path, len(rows)

    def close(self):
        self.c.close()


def put_label(img, text, x, y, color, scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 4), color, -1)
    cv2.putText(img, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 1, cv2.LINE_AA)


class ProVision:
    MODES = ["ALL", "OBJECTS", "FACES"]

    def __init__(self):
        self.db = DB(DB_PATH)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        print("[INIT] Loading YOLO", YOLO_MODEL)
        self.yolo = YOLO(YOLO_MODEL)

        print("[INIT] Loading InsightFace buffalo_l")
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        print("[INIT] Loading FER+ emotion")
        self.emo_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)

        self.mesh = None
        if HAS_MP:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=4, refine_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("[INIT] MediaPipe mesh ready")

        self.known = {}
        self.reload_faces()

        self.face_cache = {}
        self.unknown_embs = []
        self.last_person_log = {}
        self.last_unknown = 0.0
        self.mode_idx = 0
        self.mesh_on = True
        self.t0 = time.time(); self.frames = 0; self.fps = 0
        self.yolo_ms = 0.0; self.face_ms = 0.0; self.frame_id = 0

    def reload_faces(self):
        self.known = {}
        os.makedirs(KNOWN_DIR, exist_ok=True)
        for fn in os.listdir(KNOWN_DIR):
            if not fn.lower().endswith((".jpg",".jpeg",".png",".bmp")): continue
            img = cv2.imread(os.path.join(KNOWN_DIR, fn))
            if img is None: continue
            fs = self.face_app.get(img)
            if not fs:
                print("[WARN] no face in", fn); continue
            f = max(fs, key=lambda z:(z.bbox[2]-z.bbox[0])*(z.bbox[3]-z.bbox[1]))
            self.known[os.path.splitext(fn)[0]] = f.embedding
            print("[OK] loaded face:", fn)
        print("[INFO] known faces:", len(self.known))

    def recognize(self, emb):
        if not self.known: return "Unknown", 0.0
        best, bs = "Unknown", 0.0
        n = np.linalg.norm(emb)
        for nm, k in self.known.items():
            s = float(np.dot(emb, k) / (n * np.linalg.norm(k) + 1e-8))
            if s > bs:
                bs = s
                if s >= FACE_THRESHOLD: best = nm
        return best, bs

    def emotion(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (64, 64)).astype(np.float32)
        g = (g - 127.0) / 128.0
        self.emo_net.setInput(g.reshape(1, 1, 64, 64))
        s = self.emo_net.forward().flatten()
        i = int(np.argmax(s))
        return EMOTION_LABELS[i], float(s[i])

    def maybe_unknown(self, emb, crop, emo, ec):
        now = time.time()
        for e in self.unknown_embs:
            if float(np.dot(emb, e) / (np.linalg.norm(emb)*np.linalg.norm(e)+1e-8)) > 0.55:
                return
        if now - self.last_unknown < UNKNOWN_ALERT or crop.size == 0:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = os.path.join(SNAPSHOT_DIR, "unknown_" + ts + ".jpg")
        cv2.imwrite(snap, crop)
        self.db.log_unknown(snap)
        self.db.log_person("Unknown", emo, ec, 0.0, False)
        self.unknown_embs.append(emb)
        if len(self.unknown_embs) > 50: self.unknown_embs.pop(0)
        self.last_unknown = now
        print("[ALERT] Unknown ->", snap)

    def draw_mesh(self, frame):
        if not self.mesh: return frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = self.mesh.process(rgb)
        if r.multi_face_landmarks:
            for lm in r.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, lm,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=1))
        return frame

    def hud(self, frame, n_obj, n_face):
        h, w = frame.shape[:2]
        self.frames += 1
        now = time.time()
        if now - self.t0 >= 0.5:
            self.fps = self.frames / (now - self.t0); self.frames = 0; self.t0 = now
        cv2.rectangle(frame, (0, 0), (w, 44), (12, 12, 16), -1)
        txt = "FPS %d | Y %d obj %.0fms | F %d faces %.0fms | %s | mesh:%s" % (
            int(self.fps), n_obj, self.yolo_ms, n_face, self.face_ms,
            self.MODES[self.mode_idx], "ON" if (self.mesh and self.mesh_on) else "OFF")
        cv2.putText(frame, txt, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        try:
            d, a, u = self.db.stats()
            cv2.putText(frame, "Today: %d det | %d people | %d unknown" % (d, a, u),
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        except Exception:
            pass
        cv2.putText(frame, "Q quit | R faces | S shot | E csv | M mode | G mesh",
                    (w - 480, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("ERROR: camera not opening. CAMERA_ID=0, try 1"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        win = "Pro Vision"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("\n[LIVE] Q quit | M mode | G mesh\n")
        shot = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] frame read failed"); break
            frame = cv2.flip(frame, 1)
            self.frame_id += 1
            H, W = frame.shape[:2]
            mode = self.MODES[self.mode_idx]

            # --- YOLO on full frame ---
            t0 = time.time()
            try:
                results = self.yolo(frame, imgsz=IMG_SIZE, conf=YOLO_CONF,
                                    iou=YOLO_IOU, verbose=False)
                r = results[0]
            except Exception as e:
                print("[ERR] YOLO:", e); r = None
            self.yolo_ms = (time.time() - t0) * 1000.0

            n_obj = 0
            if r is not None and r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                n_obj = len(xyxy)
                names_map = self.yolo.names
                for (x1,y1,x2,y2), cf, cid in zip(xyxy, confs, clss):
                    if mode == "FACES": continue
                    nm = names_map[int(cid)]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    x1 = max(0,x1); y1 = max(0,y1)
                    x2 = min(W-1,x2); y2 = min(H-1,y2)
                    if x2 <= x1 or y2 <= y1: continue
                    color = (0, 200, 255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    put_label(frame, "%s %.2f" % (nm, cf), x1, y1, color)

            # --- InsightFace on full frame ---
            t1 = time.time()
            n_face = 0
            if mode != "OBJECTS":
                try:
                    faces = self.face_app.get(frame)
                except Exception as e:
                    print("[ERR] InsightFace:", e); faces = []
                n_face = len(faces)
                for face in faces:
                    fb = face.bbox.astype(int)
                    x1,y1,x2,y2 = max(0,fb[0]),max(0,fb[1]),min(W-1,fb[2]),min(H-1,fb[3])
                    if x2 <= x1 or y2 <= y1: continue
                    crop = frame[y1:y2, x1:x2]
                    nm, sim = self.recognize(face.embedding)
                    emo, ec = ("-", 0.0)
                    if crop.size > 0:
                        try: emo, ec = self.emotion(crop)
                        except Exception: pass
                    col = (0, 255, 0) if nm != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                    put_label(frame, "%s | %s %.0f%%" % (nm, emo, ec*100),
                              x1, y1, col, 0.55)
                    now = time.time()
                    if nm != "Unknown":
                        if now - self.last_person_log.get(nm, 0.0) >= LOG_COOLDOWN:
                            self.db.log_person(nm, emo, ec, sim, True)
                            self.last_person_log[nm] = now
                            print("[FACE] %s | %s | sim=%.2f" % (nm, emo, sim))
                    else:
                        self.maybe_unknown(face.embedding, crop, emo, ec)
            self.face_ms = (time.time() - t1) * 1000.0

            # --- Mesh ---
            if self.mesh and self.mesh_on and mode != "OBJECTS":
                try: frame = self.draw_mesh(frame)
                except Exception as e: print("[ERR] mesh:", e)

            # --- Debug print every second ---
            if self.frame_id % 30 == 0:
                print("[DBG] frame %d | objects=%d faces=%d | known=%d" % (
                    self.frame_id, n_obj, n_face, len(self.known)))

            frame = self.hud(frame, n_obj, n_face)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q"), 27): break
            elif k in (ord("r"), ord("R")): self.reload_faces()
            elif k in (ord("s"), ord("S")):
                shot += 1; fn = "screenshot_%d.png" % shot
                cv2.imwrite(fn, frame); print("[SHOT]", fn)
            elif k in (ord("e"), ord("E")):
                p, n = self.db.export_csv(); print("[EXPORT]", p, n)
            elif k in (ord("m"), ord("M")):
                self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
                print("[MODE]", self.MODES[self.mode_idx])
            elif k in (ord("g"), ord("G")):
                self.mesh_on = not self.mesh_on
                print("[MESH]", self.mesh_on)

        cap.release(); cv2.destroyAllWindows()
        p, n = self.db.export_csv(); print("[EXPORT]", p, n)
        self.db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  PRO VISION - objects + faces + emotion + green mesh")
    print("=" * 60)
    print("  YOLO:    %s @ %dpx" % (YOLO_MODEL, IMG_SIZE))
    print("  Face:    InsightFace ArcFace")
    print("  Emotion: FER+ ONNX")
    print("  Mesh:    MediaPipe FACEMESH_TESSELATION")
    print("  Faces:   " + os.path.abspath(KNOWN_DIR))
    print("=" * 60)
    ProVision().run()
