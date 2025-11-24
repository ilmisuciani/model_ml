# main.py
from typing import Optional, List, Dict, Any

from pathlib import Path
import pandas as pd
import pickle

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Folder tempat main.py berada
BASE_DIR = Path(__file__).resolve().parent
print("BASE_DIR:", BASE_DIR)  # optional, buat cek di terminal

# =========================================================
# Inisialisasi Aplikasi FastAPI
# =========================================================
app = FastAPI(
    title="AI Learning Insight API",
    description="API untuk menghasilkan AI Learning Insight "
                "berdasarkan perilaku belajar siswa di platform Dicoding.",
    version="1.0.0",
)

# =========================================================
# Load Artifacts (Model, Scaler, Data Clustered)
# =========================================================

try:
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(BASE_DIR / "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    df_clustered = pd.read_csv(BASE_DIR / "clustered_students.csv")

    REQUIRED_COLS = [
        "developer_id",
        "developer_name",
        "cluster_label",
        "total_active_days",
        "avg_completion_time_hours",
        "total_journeys_completed",
        "rejection_ratio",
        "avg_exam_score",
    ]
    missing_cols = [c for c in REQUIRED_COLS if c not in df_clustered.columns]
    if missing_cols:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di clustered_students.csv: {missing_cols}"
        )

    df_clustered["cluster_label"] = df_clustered["cluster_label"].astype(int)

except Exception as e:
    print("Gagal memuat artifacts model:", e)
    scaler = None
    kmeans = None
    df_clustered = None

# =========================================================
# DEFINISI CLUSTER PROFILES (nama cluster)
# =========================================================

CLUSTER_PROFILES: Dict[int, Dict[str, Any]] = {
    0: {
        "label_id": "Fast Learner",
        "short_description": (
            "Aktivitas belajar masih jarang, tetapi ketika mulai belajar mampu menyelesaikan modul dengan sangat cepat "
            "dan mempertahankan nilai ujian yang cukup baik. Volume journey relatif rendah dan hampir tidak ada revisi submission."
        ),
        "concept_tag": "fast_learner",
    },
    1: {
        "label_id": "Consistent Learner",
        "short_description": (
            "Belajar secara konsisten, menyelesaikan banyak journey, dan memiliki nilai ujian yang tinggi. "
            "Tingkat refleksi berada pada kisaran sedang."
        ),
        "concept_tag": "consistent_learner",
    },
    2: {
        "label_id": "Reflective Learner",
        "short_description": (
            "Sangat sering aktif dan menyelesaikan banyak journey, namun membutuhkan waktu yang panjang per modul. "
            "Cenderung mengulas materi secara mendalam."
        ),
        "concept_tag": "reflective_learner",
    },
    3: {
        "label_id": "Struggling Learner",
        "short_description": (
            "Cukup aktif dan banyak bereksperimen dengan submission (revisi tinggi), "
            "namun nilai ujian relatif rendah sehingga masih perlu penguatan konsep."
        ),
        "concept_tag": "struggling_learner",
    },
}

# =========================================================
# TEMPLATE KALIMAT INSIGHT UNTUK SETIAP CLUSTER
# =========================================================

CLUSTER_TEMPLATES: Dict[int, str] = {
    0: (
        "Aktivitas belajarmu masih jarang (sekitar {active_days:.0f} hari aktif), "
        "tetapi ketika mulai belajar kamu bergerak sangat cepat dengan rata-rata waktu selesai "
        "sekitar {avg_time_hours:.1f} jam per modul. Kamu telah menyelesaikan sekitar {journeys:.0f} journey "
        "dengan nilai ujian rata-rata {score:.0f}. Cobalah meningkatkan frekuensi belajar agar dampak "
        "pembelajaranmu lebih konsisten."
    ),
    1: (
        "Kamu belajar secara cukup konsisten (sekitar {active_days:.0f} hari aktif) dan telah menyelesaikan "
        "sekitar {journeys:.0f} journey. Nilai ujian rata-ratamu tinggi, yaitu sekitar {score:.0f}. "
        "Tingkat refleksi melalui submission yang ditolak berada di kisaran {rejection_ratio:.2f}. "
        "Pertahankan pola belajar ini dan gunakan umpan balik untuk terus menyempurnakan pemahamanmu."
    ),
    2: (
        "Kamu sangat tekun dengan sekitar {active_days:.0f} hari aktif dan telah menyelesaikan "
        "sekitar {journeys:.0f} journey. Rata-rata waktu yang kamu habiskan per modul cukup panjang, "
        "sekitar {avg_time_hours:.1f} jam. Nilai ujian rata-ratamu sekitar {score:.0f}. "
        "Pertahankan kedalaman belajarmu, namun pertimbangkan pengelolaan waktu belajar yang lebih efisien."
    ),
    3: (
        "Kamu cukup aktif belajar (sekitar {active_days:.0f} hari aktif) dan telah menyelesaikan "
        "sekitar {journeys:.0f} journey. Rata-rata nilai ujianmu saat ini sekitar {score:.0f}, "
        "dengan rasio submission ditolak sekitar {rejection_ratio:.2f}. Ini menunjukkan kamu banyak "
        "bereksperimen, tetapi masih perlu memperkuat pemahaman konsep dasar. Manfaatkan kembali materi, "
        "contoh solusi, dan umpan balik dari submission untuk meningkatkan hasil ujian."
    ),
}

# =========================================================
# Pydantic Models (Schema untuk request & response)
# =========================================================

class InsightResponse(BaseModel):
    developer_id: int
    developer_name: str
    cluster_id: int
    cluster_label: str          # pakai label_id saja
    concept_tag: Optional[str]
    short_description: str
    insight_text: str


class ClusterProfileResponse(BaseModel):
    cluster_id: int
    label_id: str
    concept_tag: Optional[str]
    short_description: str


class PredictRequest(BaseModel):
    total_active_days: float
    avg_completion_time_hours: float
    total_journeys_completed: float
    rejection_ratio: float
    avg_exam_score: float
    developer_id: Optional[int] = None
    developer_name: Optional[str] = None


class PredictResponse(BaseModel):
    developer_id: Optional[int]
    developer_name: Optional[str]
    cluster_id: int
    cluster_label: str
    concept_tag: Optional[str]
    short_description: str
    insight_text: str

# =========================================================
# Helper: Generate Insight dari satu baris data
# =========================================================

def generate_insight_for_row(row: pd.Series) -> Dict[str, Any]:
    cluster_id = int(row["cluster_label"])
    profile = CLUSTER_PROFILES.get(cluster_id, {})
    template = CLUSTER_TEMPLATES.get(cluster_id, "")

    insight_text = template.format(
        active_days=row["total_active_days"],
        avg_time_hours=row["avg_completion_time_hours"],
        journeys=row["total_journeys_completed"],
        rejection_ratio=row["rejection_ratio"],
        score=row["avg_exam_score"],
    )

    return {
        "developer_id": int(row["developer_id"]),
        "developer_name": str(row["developer_name"]),
        "cluster_id": cluster_id,
        "cluster_label": profile.get("label_id", f"Cluster {cluster_id}"),
        "concept_tag": profile.get("concept_tag"),
        "short_description": profile.get("short_description", ""),
        "insight_text": insight_text,
    }

# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/health", tags=["system"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": kmeans is not None and scaler is not None,
        "data_loaded": df_clustered is not None,
    }


@app.get("/clusters", response_model=List[ClusterProfileResponse], tags=["clusters"])
def list_clusters():
    profiles = []
    for cid, meta in CLUSTER_PROFILES.items():
        profiles.append(
            ClusterProfileResponse(
                cluster_id=cid,
                label_id=meta["label_id"],
                concept_tag=meta.get("concept_tag"),
                short_description=meta["short_description"],
            )
        )
    return profiles


@app.get("/insights/{developer_id}", response_model=InsightResponse, tags=["insights"])
def get_insight_by_developer_id(developer_id: int):
    if df_clustered is None:
        raise HTTPException(status_code=500, detail="Data clustering belum ter-load.")

    row = df_clustered[df_clustered["developer_id"] == developer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Developer ID tidak ditemukan di data clustering.")

    insight_dict = generate_insight_for_row(row.iloc[0])
    return InsightResponse(**insight_dict)


@app.get("/insights", response_model=List[InsightResponse], tags=["insights"])
def list_insights(limit: int = Query(10, ge=1, le=100)):
    if df_clustered is None:
        raise HTTPException(status_code=500, detail="Data clustering belum ter-load.")

    subset = df_clustered.head(limit)
    insights = [generate_insight_for_row(row) for _, row in subset.iterrows()]
    return [InsightResponse(**ins_) for ins_ in insights]


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict_cluster(payload: PredictRequest):
    if scaler is None or kmeans is None:
        raise HTTPException(status_code=500, detail="Model belum ter-load.")

    X = [[
        payload.total_active_days,
        payload.avg_completion_time_hours,
        payload.total_journeys_completed,
        payload.rejection_ratio,
        payload.avg_exam_score,
    ]]

    X_scaled = scaler.transform(X)
    cluster_id = int(kmeans.predict(X_scaled)[0])

    profile = CLUSTER_PROFILES.get(cluster_id, {})
    template = CLUSTER_TEMPLATES.get(cluster_id, "")

    insight_text = template.format(
        active_days=payload.total_active_days,
        avg_time_hours=payload.avg_completion_time_hours,
        journeys=payload.total_journeys_completed,
        rejection_ratio=payload.rejection_ratio,
        score=payload.avg_exam_score,
    )

    return PredictResponse(
        developer_id=payload.developer_id,
        developer_name=payload.developer_name or "Unknown",
        cluster_id=cluster_id,
        cluster_label=profile.get("label_id", f"Cluster {cluster_id}"),
        concept_tag=profile.get("concept_tag"),
        short_description=profile.get("short_description", ""),
        insight_text=insight_text,
    )

# =========================================================
# Cara menjalankan:
#   uvicorn main:app --reload
# =========================================================
