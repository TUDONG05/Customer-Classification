import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans as SKMeans
import os
BASE = os.path.dirname(os.path.abspath(__file__))
MALL_PATH   = os.path.join(BASE, "Mall_Customers.csv")
RETAIL_PATH = os.path.join(BASE, "Online Retail.xlsx")

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def nearest_centroid_predict(centroids, cluster_ids, X):
    """Gán mỗi điểm trong X vào cụm có centroid gần nhất."""
    result = []
    for x in X:
        dists = [euclidean(x, c) for c in centroids]
        result.append(cluster_ids[int(np.argmin(dists))])
    return np.array(result)

def dbscan_predict_core(X_new, X_core, core_labels, eps):
    """
    điểm mới được gán vào cụm của core point gần nhất nếu khoảng cách <= eps, ngược lại trả về -1 (noise).
    """
    results = []
    for x in X_new:
        dists  = np.sqrt(((X_core - x) ** 2).sum(axis=1))
        within = np.where(dists <= eps)[0]
        if len(within) == 0:
            results.append(-1)
        else:
            nearest = within[int(np.argmin(dists[within]))]
            results.append(int(core_labels[nearest]))
    return np.array(results)

class KMeans:
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.centroids_   = None
        self.labels_      = None

    def _init(self, X):
        rng = np.random.RandomState(self.random_state)
        return X[rng.choice(len(X), self.n_clusters, replace=False)].copy()

    def _assign(self, X):
        return np.array([
            np.argmin([euclidean(x, c) for c in self.centroids_])
            for x in X
        ])

    def fit(self, X):
        self.centroids_ = self._init(X)
        for _ in range(self.max_iter):
            labels = self._assign(X)
            new_c  = np.array([
                X[labels == k].mean(0) if (labels == k).any() else self.centroids_[k]
                for k in range(self.n_clusters)
            ])
            shift = max(euclidean(self.centroids_[k], new_c[k]) for k in range(self.n_clusters))
            self.centroids_ = new_c
            if shift < self.tol:
                break
        self.labels_ = self._assign(X)
        return self

    def predict(self, X):
        return self._assign(X)


#  Custom Agglomerative (Ward) 
class Agglomerative:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.labels_    = None
        self.centroids_ = None

    @staticmethod
    def _ward(ci, cj):
        ni, nj = len(ci), len(cj)
        return (ni * nj) / (ni + nj) * np.sum((ci.mean(0) - cj.mean(0)) ** 2)

    def fit_predict(self, X):
        clusters       = {i: X[i:i+1].copy() for i in range(len(X))}
        point_cluster  = np.arange(len(X))
        next_id        = len(X)

        while len(clusters) > self.n_clusters:
            ids  = list(clusters.keys())
            best = (np.inf, ids[0], ids[1])
            for p in range(len(ids)):
                for q in range(p + 1, len(ids)):
                    d = self._ward(clusters[ids[p]], clusters[ids[q]])
                    if d < best[0]:
                        best = (d, ids[p], ids[q])
            _, i, j = best
            merged = np.vstack([clusters[i], clusters[j]])
            clusters[next_id] = merged
            point_cluster[point_cluster == i] = next_id
            point_cluster[point_cluster == j] = next_id
            del clusters[i], clusters[j]
            next_id += 1

        final_ids    = list(clusters.keys())
        id_map       = {fid: idx for idx, fid in enumerate(final_ids)}
        self.labels_ = np.array([id_map[point_cluster[i]] for i in range(len(X))])

        unique          = np.unique(self.labels_)
        self.centroids_ = np.array([X[self.labels_ == k].mean(0) for k in unique])
        return self.labels_

    def predict(self, X):
        return nearest_centroid_predict(self.centroids_, list(range(self.n_clusters)), X)


#Custom DBSCAN 
class DBSCANScratch:
    def __init__(self, eps=10, min_samples=3):
        self.eps         = eps
        self.min_samples = min_samples
        self.labels_     = None
        self.centroids_  = None
        self._cids       = None

    def _neighbors(self, X, i):
        return [j for j in range(len(X)) if euclidean(X[i], X[j]) <= self.eps]

    def fit_predict(self, X):
        n      = len(X)
        labels = np.full(n, -1, dtype=int)
        cid    = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            nb = self._neighbors(X, i)
            if len(nb) < self.min_samples:
                continue
            labels[i] = cid
            queue = list(nb)
            while queue:
                q = queue.pop(0)
                if labels[q] == -1:
                    labels[q] = cid
                if labels[q] != cid:
                    continue
                labels[q] = cid
                qnb = self._neighbors(X, q)
                if len(qnb) >= self.min_samples:
                    queue.extend(p for p in qnb if labels[p] == -1)
            cid += 1
        self.labels_ = labels
        self._cids   = [k for k in np.unique(labels) if k != -1]

        # Centroids dùng để đặt tên cụm
        if self._cids:
            self.centroids_ = np.array([X[labels == k].mean(0) for k in self._cids])

        # Core points dùng để predict đúng bản chất DBSCAN (eps-reachability)
        core_mask         = np.array([len(self._neighbors(X, i)) >= self.min_samples for i in range(n)])
        self._X_core      = X[core_mask].copy()
        self._core_labels = labels[core_mask].copy()
        return labels

    def predict(self, X):
        if len(self._X_core) == 0:
            return np.full(len(X), -1)
        # Gán vào cụm của core point gần nhất trong eps, không tìm được → noise
        return dbscan_predict_core(X, self._X_core, self._core_labels, self.eps)


# Đặt tên cụm Mall — theo vị trí tâm cụm (đơn vị gốc) so với median toàn bộ
def _name_mall(inc, sco, inc_med, sco_med):
    if abs(inc - inc_med) < 18 and abs(sco - sco_med) < 18:
        return "Khách hàng Ổn định"
    if inc < inc_med and sco < sco_med:
        return "Khách hàng Phổ thông"
    if inc < inc_med and sco >= sco_med:
        return "Khách hàng Trải nghiệm"
    if inc >= inc_med and sco < sco_med:
        return "Khách hàng Thận trọng"
    return "Khách hàng Cao cấp"


def _disambiguate_names(name_map):
    """_name_mall chỉ có 5 tên theo góc phần tư, nhưng số cụm thực tế (Agglomerative
    K=3, DBSCAN K=7) có thể khác 5 → nhiều cụm rơi vào cùng góc phần tư sẽ trùng tên.
    Đánh số thêm (1, 2, ...) cho các tên bị trùng để phân biệt, giống notebook."""
    counts = {}
    for name in name_map.values():
        counts[name] = counts.get(name, 0) + 1
    seen, result = {}, {}
    for k, name in name_map.items():
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            result[k] = f"{name} {seen[name]}"
        else:
            result[k] = name
    return result


# Load & fit Mall models
print("Đang tải Mall_Customers và fit 3 mô hình...")
_mall_df = pd.read_csv(MALL_PATH)
X3       = _mall_df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# Chuẩn hóa (StandardScaler) trước khi phân cụm — như notebook, tránh Annual Income
# (thang chục) lấn át Spending Score (thang trăm) khi tính khoảng cách Euclidean.
_mall_scaler = StandardScaler().fit(X3)
X3_scaled    = _mall_scaler.transform(X3)

_MALL_AGG_K  = 3     # K tự chọn từ dendrogram (notebook: distance_threshold=83.3664)
_MALL_DB_EPS = 0.35  # trên dữ liệu đã chuẩn hóa (notebook)

_km_mall = KMeans(n_clusters=5, random_state=42).fit(X3_scaled)

_agg_mall        = Agglomerative(n_clusters=_MALL_AGG_K)
_agg_mall_labels = _agg_mall.fit_predict(X3_scaled)

_db_mall        = DBSCANScratch(eps=_MALL_DB_EPS, min_samples=3)
_db_mall_labels = _db_mall.fit_predict(X3_scaled)

# Tâm cụm ở đơn vị GỐC (income k$, score 1-100) chỉ dùng để đặt tên cho dễ diễn giải —
# tách biệt với centroids_/core points nội bộ (đang ở không gian đã chuẩn hóa, dùng để predict).
_inc_med, _sco_med = np.median(X3[:, 0]), np.median(X3[:, 1])

_km_mall_raw_c  = {k: X3[_km_mall.labels_ == k].mean(axis=0) for k in range(5)}
_agg_mall_raw_c = {k: X3[_agg_mall_labels == k].mean(axis=0) for k in np.unique(_agg_mall_labels)}
_db_mall_raw_c  = {k: X3[_db_mall_labels == k].mean(axis=0) for k in _db_mall._cids}

_km_mall_names  = _disambiguate_names({k: _name_mall(*c, _inc_med, _sco_med) for k, c in _km_mall_raw_c.items()})
_agg_mall_names = _disambiguate_names({k: _name_mall(*c, _inc_med, _sco_med) for k, c in _agg_mall_raw_c.items()})
_db_mall_names  = _disambiguate_names({k: _name_mall(*c, _inc_med, _sco_med) for k, c in _db_mall_raw_c.items()})

print("  Mall models OK")


def predict_mall(income: float, score: float, algo: str) -> str:
    pt    = np.array([[income, score]])
    pt_sc = _mall_scaler.transform(pt)
    if algo == "K-Means":
        lbl  = int(_km_mall.predict(pt_sc)[0])
        name = _km_mall_names.get(lbl, f"Cụm {lbl}")
    elif algo == "Agglomerative":
        lbl  = int(_agg_mall.predict(pt_sc)[0])
        name = _agg_mall_names.get(lbl, f"Cụm {lbl}")
    else:  # DBSCAN
        lbl  = int(_db_mall.predict(pt_sc)[0])
        if lbl == -1:
            return "**Điểm ngoại lệ (Noise)**\n\nKhách hàng này không thuộc bất kỳ cụm nào — giá trị thu nhập hoặc chi tiêu quá khác biệt so với các nhóm đã học. DBSCAN không ép điểm vào cụm gần nhất như K-Means."
        name = _db_mall_names.get(lbl, f"Cụm {lbl}")

    # tên có thể bị _disambiguate_names thêm hậu tố số (vd "Phổ thông 1") — tra mô tả theo tên gốc
    desc_parts = name.rsplit(" ", 1)
    desc_key   = desc_parts[0] if len(desc_parts) == 2 and desc_parts[1].isdigit() else name

    desc = {
        "Khách hàng Phổ thông":   "Thu nhập thấp, chi tiêu thấp — hạn chế tài chính.",
        "Khách hàng Trải nghiệm": "Thu nhập thấp nhưng sẵn sàng chi tiêu trải nghiệm.",
        "Khách hàng Ổn định":     "Thu nhập & chi tiêu ở mức trung bình, cân bằng.",
        "Khách hàng Thận trọng":  "Thu nhập cao nhưng chi tiêu có chọn lọc.",
        "Khách hàng Cao cấp":     "Thu nhập cao, chi tiêu lớn — nhóm mục tiêu chiến lược.",
    }
    return f"**{name}**\n\n{desc.get(desc_key, '')}"



#Online Retail RFM (sklearn) 
print("Đang tải Online Retail.xlsx và xây dựng RFM (vài giây)...")
_retail_raw = pd.read_excel(RETAIL_PATH, sheet_name="Online Retail")
_retail = _retail_raw.dropna(subset=["CustomerID"]).copy()
_retail["CustomerID"] = _retail["CustomerID"].astype(int)
_retail = _retail[~_retail["InvoiceNo"].astype(str).str.startswith("C")]
_retail = _retail[(_retail["Quantity"] > 0) & (_retail["UnitPrice"] > 0)]
_retail["Revenue"] = _retail["Quantity"] * _retail["UnitPrice"]

_snap = _retail["InvoiceDate"].max() + pd.Timedelta(days=1)
_rfm  = _retail.groupby("CustomerID").agg(
    Recency   = ("InvoiceDate", lambda x: (_snap - x.max()).days),
    Frequency = ("InvoiceNo",   "nunique"),
    Monetary  = ("Revenue",     "sum"),
).reset_index()

# MinMaxScaler (KMeans + Agg) — trên RFM gốc, KHÔNG log (đúng như notebook)
_mm_scaler  = MinMaxScaler().fit(_rfm[["Recency", "Frequency", "Monetary"]])
X_mm        = _mm_scaler.transform(_rfm[["Recency", "Frequency", "Monetary"]])

# StandardScaler (DBSCAN) — trên RFM gốc
_std_scaler = StandardScaler().fit(_rfm[["Recency", "Frequency", "Monetary"]])
X_std       = _std_scaler.transform(_rfm[["Recency", "Frequency", "Monetary"]])

# Fit models — K-Means dùng sklearn thật (k-means++, n_init=10) như notebook,
# không phải class KMeans tự cài (chỉ dùng cho tab Mall Customers)
_RFM_K = 3  # K_OPTIMAL từ Elbow (notebook)
_AGG_K = 2  # AGG_CLUSTERS từ dendrogram (notebook, distance_threshold=15.8055)

_km_rfm  = SKMeans(n_clusters=_RFM_K, init="k-means++", random_state=42, n_init=10).fit(X_mm)
_agg_rfm = AgglomerativeClustering(n_clusters=_AGG_K, metric="euclidean", linkage="ward")
_agg_rfm_labels = _agg_rfm.fit_predict(X_mm)
_db_rfm  = DBSCAN(eps=0.3, min_samples=5).fit(X_std)

# Centroids cho Agg (sklearn không có predict)
_rfm["_km"]  = _km_rfm.labels_
_rfm["_agg"] = _agg_rfm_labels
_rfm["_db"]  = _db_rfm.labels_

_agg_centroids = np.array([X_mm[_agg_rfm_labels == k].mean(0) for k in range(_AGG_K)])

# DBSCAN: lưu core points để predict đúng bản chất (eps-reachability)
_DB_EPS = 0.3
_db_core_idx    = _db_rfm.core_sample_indices_
_db_X_core      = X_std[_db_core_idx]
_db_core_labels = _db_rfm.labels_[_db_core_idx]


def _auto_name_rfm(rfm_df, cluster_col):
    """Đặt tên cụm dựa trên ranking RFM tổng hợp."""
    valid = rfm_df[rfm_df[cluster_col] != -1].copy()
    means = valid.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()
    norm  = (means - means.min()) / (means.max() - means.min() + 1e-9)
    means["_score"] = -norm["Recency"] + norm["Frequency"] + norm["Monetary"]
    ranked = means["_score"].rank(ascending=False).astype(int)
    pool   = {
        4: ["Khách VIP", "Khách tiềm năng", "Khách không hoạt động", "Khách rời bỏ"],
        3: ["Khách VIP", "Khách tiềm năng", "Khách rời bỏ"],
        2: ["Khách hoạt động", "Khách ít hoạt động"],
    }
    labels = pool.get(len(means), [f"Phân khúc {i+1}" for i in range(len(means))])
    name_map = {int(cid): labels[rank - 1] for cid, rank in ranked.items()}
    if -1 in rfm_df[cluster_col].unique():
        name_map[-1] = "Ngoại lệ"
    return name_map

_km_rfm_names  = _auto_name_rfm(_rfm, "_km")
_agg_rfm_names = _auto_name_rfm(_rfm, "_agg")
_db_rfm_names  = _auto_name_rfm(_rfm, "_db")
print("  Online Retail models OK")


def predict_rfm(recency: float, frequency: float, monetary: float, algo: str) -> str:
    raw    = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
    pt_mm  = _mm_scaler.transform(raw.values)
    pt_std = _std_scaler.transform(raw.values)

    if algo == "K-Means":
        lbl  = int(_km_rfm.predict(pt_mm)[0])
        name = _km_rfm_names.get(lbl, f"Cụm {lbl}")
    elif algo == "Agglomerative":
        lbl  = int(nearest_centroid_predict(_agg_centroids, list(range(_AGG_K)), pt_mm)[0])
        name = _agg_rfm_names.get(lbl, f"Cụm {lbl}")
    else:  # DBSCAN
        lbl = int(dbscan_predict_core(pt_std, _db_X_core, _db_core_labels, _DB_EPS)[0])
        name = _db_rfm_names.get(lbl, "Ngoại lệ")

    desc = {
        "Khách VIP":             "Mua gần đây, thường xuyên, chi tiêu lớn — ưu tiên giữ chân.",
        "Khách tiềm năng":       "Mua gần đây nhưng chưa đủ gắn bó — cần nurturing.",
        "Khách không hoạt động": "Từng có giá trị nhưng đang ngủ đông — tái kích hoạt.",
        "Khách rời bỏ":          "Lâu không quay lại, chi tiêu thấp — nguy cơ mất vĩnh viễn.",
        "Khách hoạt động":       "Vẫn mua đều, giá trị ở mức trung bình.",
        "Khách ít hoạt động":    "Ít mua, giá trị thấp — cần chương trình kích hoạt.",
        "Ngoại lệ":              "Điểm dữ liệu bất thường, không thuộc cụm nào rõ ràng.",
    }
    return f"**{name}**\n\n{desc.get(name, '')}"

#  Gradio UI 
with gr.Blocks(title="Phân Khúc Khách Hàng") as demo:
    gr.Markdown("# Phân Khúc Khách Hàng\nNhập thông tin khách hàng để nhận kết quả phân khúc.")

    with gr.Tabs():
        #Mall Customers 
        with gr.TabItem("Mall Customers"):
            gr.Markdown(
                "Dữ liệu: **Mall Customers** (200 khách hàng)  \n"
                "Phân cụm dựa trên **Annual Income** và **Spending Score**  \n"
                "Model được cài đặt từ đầu: K-Means, Agglomerative, DBSCAN"
            )
            with gr.Row():
                with gr.Column():
                    mall_income = gr.Slider(15, 137, value=60, step=1, label="Annual Income (k$)")
                    mall_score  = gr.Slider(1, 100, value=50, step=1, label="Spending Score (1-100)")
                    mall_algo   = gr.Radio(
                        ["K-Means", "Agglomerative", "DBSCAN"],
                        value="K-Means",
                        label="Thuật toán"
                    )
                    mall_btn    = gr.Button("Phân khúc", variant="primary")
                with gr.Column():
                    mall_out = gr.Markdown(label="Kết quả")

            mall_btn.click(
                predict_mall,
                inputs=[mall_income, mall_score, mall_algo],
                outputs=mall_out,
            )

            gr.Examples(
                examples=[
                    [90, 17, "K-Means"],
                    [15, 81, "K-Means"],
                    [60, 50, "Agglomerative"],
                    [86, 82, "DBSCAN"],
                ],
                inputs=[mall_income, mall_score, mall_algo],
                label="Ví dụ nhanh",
            )

        #Online Retail RFM 
        with gr.TabItem("Online Retail (RFM)"):
            gr.Markdown(
                "Dữ liệu: **Online Retail** (4 338 khách hàng)  \n"
                "Phân cụm dựa trên đặc trưng **RFM** (Recency – Frequency – Monetary)  \n"
                "Sử dụng sklearn: K-Means, Agglomerative, DBSCAN"
            )
            with gr.Row():
                with gr.Column():
                    rfm_r    = gr.Number(value=30,   label="Recency (ngày kể từ lần mua cuối)")
                    rfm_f    = gr.Number(value=5,    label="Frequency (số hóa đơn)")
                    rfm_m    = gr.Number(value=1500, label="Monetary (tổng doanh thu £)")
                    rfm_algo = gr.Radio(
                        ["K-Means", "Agglomerative", "DBSCAN"],
                        value="K-Means",
                        label="Thuật toán"
                    )
                    rfm_btn  = gr.Button("Phân khúc", variant="primary")
                with gr.Column():
                    rfm_out  = gr.Markdown(label="Kết quả")

            rfm_btn.click(
                predict_rfm,
                inputs=[rfm_r, rfm_f, rfm_m, rfm_algo],
                outputs=rfm_out,
            )

            gr.Examples(
                examples=[
                    [10,  15, 5000,  "K-Means"],
                    [300, 1,  200,   "K-Means"],
                    [50,  3,  800,   "Agglomerative"],
                    [20,  8,  3000,  "DBSCAN"],
                ],
                inputs=[rfm_r, rfm_f, rfm_m, rfm_algo],
                label="Ví dụ nhanh",
            )

if __name__ == "__main__":
    demo.launch()
 