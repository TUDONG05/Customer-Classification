# Phân Khúc Khách Hàng — Customer Segmentation

Triển khai ba thuật toán phân cụm **K-Means**, **DBSCAN**, **Agglomerative Hierarchical** trên hai bộ dữ liệu: **Mall Customers** và **Online Retail**.

---

## Bộ dữ liệu

### Mall_Customers.csv — 200 khách hàng, 5 thuộc tính

| Cột | Mô tả |
|-----|-------|
| `CustomerID` | Mã khách hàng |
| `Gender` | Giới tính |
| `Age` | Tuổi |
| `Annual Income (k$)` | Thu nhập hàng năm (nghìn USD) |
| `Spending Score (1-100)` | Điểm chi tiêu tại trung tâm thương mại |

Các thuật toán phân cụm sử dụng hai đặc trưng `Annual Income (k$)` và `Spending Score (1-100)`.

### Online Retail.xlsx — giao dịch thực tế tại Anh (12/2010 – 12/2011)

| Cột | Mô tả |
|-----|-------|
| `InvoiceNo` | Mã hóa đơn (bắt đầu bằng "C" = hóa đơn hủy) |
| `StockCode` | Mã sản phẩm |
| `Description` | Tên sản phẩm |
| `Quantity` | Số lượng |
| `InvoiceDate` | Ngày giờ hóa đơn |
| `UnitPrice` | Đơn giá (£) |
| `CustomerID` | Mã khách hàng (24.9% null) |
| `Country` | Quốc gia |

**Sau tiền xử lý:** 397,884 dòng — **4,338 khách hàng** duy nhất.

#### Đặc trưng RFM

| Đặc trưng | Ý nghĩa | Thống kê |
|-----------|---------|---------|
| **Recency (R)** | Số ngày kể từ lần mua cuối | TB: 92.5 ngày · max: 374 ngày |
| **Frequency (F)** | Số hóa đơn khác nhau | TB: 4.3 · max: 209 |
| **Monetary (M)** | Tổng doanh thu (£) | TB: £2,054 · max: £280,206 |

Quy trình chuẩn hóa theo từng thuật toán (theo phương pháp bài báo):
- **K-Means & Agglomerative:** `MinMaxScaler` trên RFM gốc → đưa giá trị về [0, 1]
- **DBSCAN:** `StandardScaler` trên RFM gốc → mean=0, std=1

---

## Thuật toán

### K-Means
- Khởi tạo K tâm cụm ngẫu nhiên
- Gán mỗi điểm vào tâm gần nhất (khoảng cách Euclidean)
- Cập nhật tâm = trung bình các điểm trong cụm
- Lặp đến hội tụ (tâm không dịch chuyển quá `tol`)
- Chọn k tối ưu qua **Elbow Method** (đạo hàm bậc 2 SSE)

### Agglomerative Hierarchical Clustering (Bottom-up)
- Bắt đầu: mỗi điểm là một cụm riêng
- Tại mỗi bước gộp cặp cụm có khoảng cách nhỏ nhất
- Hỗ trợ 4 linkage: `ward`, `single`, `complete`, `average`
- Ward linkage: tối thiểu hóa tăng phương sai khi gộp
- Kết quả trực quan qua **Dendrogram**

### DBSCAN
- Xác định **core point**: điểm có ≥ `min_samples` láng giềng trong bán kính `eps`
- Mở rộng cụm từ core point theo BFS
- Điểm không thuộc cụm nào → **noise** (nhãn -1)
- Chọn `eps` qua **k-distance graph** (điểm khuỷu tay)

---

## Tiêu chí đặt tên cụm

### Mall Customers

Tên cụm được xác định bằng cách so sánh **tọa độ tâm cụm** với **giá trị median** của toàn bộ dữ liệu:

> Median Income ≈ 61.5 k$ · Median Spending Score ≈ 50

| Tên | Thu nhập | Điểm chi tiêu | Điều kiện | Ý nghĩa |
|-----|----------|---------------|-----------|---------|
| **Khách hàng Ổn định** | Trung bình | Trung bình | \|Income − 61.5\| < 18 **và** \|Score − 50\| < 18 | Nhóm trung lưu, hành vi cân bằng |
| **Khách hàng Phổ thông** | Thấp | Thấp | Income < 61.5 **và** Score < 50 | Hạn chế tài chính, chi tiêu cẩn trọng |
| **Khách hàng Trải nghiệm** | Thấp | Cao | Income < 61.5 **và** Score ≥ 50 | Thu nhập thấp nhưng sẵn sàng chi tiêu |
| **Khách hàng Thận trọng** | Cao | Thấp | Income ≥ 61.5 **và** Score < 50 | Thu nhập cao, chi tiêu có chọn lọc |
| **Khách hàng Cao cấp** | Cao | Cao | Income ≥ 61.5 **và** Score ≥ 50 | Nhóm mục tiêu — VIP |

> DBSCAN bổ sung nhãn **Điểm ngoại lai** (nhãn -1) cho các khách hàng không thuộc cụm nào.

### Online Retail (RFM)

Tên cụm được xác định bằng **điểm tổng hợp RFM** tính trên giá trị trung bình mỗi cụm:

```
Score = −norm(Recency) + norm(Frequency) + norm(Monetary)
```

- `norm(x)` = chuẩn hóa min-max trong khoảng [0, 1]
- **Recency thấp** (mua gần đây) → đóng góp dương vào score
- **Frequency và Monetary cao** → đóng góp dương vào score
- Cụm có **score cao nhất** → tên tốt nhất (Khách VIP)

**Bảng xếp hạng → tên cụm:**

| Hạng score | Tên (3 cụm) | Tên (2 cụm) |
|------------|-------------|-------------|
| 1 (cao nhất) | **Khách VIP** | **Khách hoạt động** |
| 2 | **Khách tiềm năng** | **Khách ít hoạt động** |
| 3 (thấp nhất) | **Khách rời bỏ** | — |

**Đặc điểm RFM và chiến lược từng phân khúc:**

| Phân khúc | Recency | Frequency | Monetary | Lý do đặt tên | Chiến lược đề xuất |
|-----------|---------|-----------|----------|---------------|--------------------|
| **Khách VIP** | Thấp nhất | Cao nhất | Cao nhất | R thấp → mua rất gần đây; F & M cao nhất → thường xuyên & chi tiêu lớn | Loyalty program, ưu đãi độc quyền, giữ chân ưu tiên |
| **Khách tiềm năng** | Thấp | Trung bình | Trung bình | R thấp → vẫn mua gần đây; F & M trung bình → chưa đủ gắn bó | Welcome offer, gợi ý sản phẩm phù hợp, nurturing |
| **Khách rời bỏ** | Cao nhất | Thấp nhất | Thấp nhất | R rất cao → rất lâu không quay lại; F & M thấp nhất → nguy cơ mất vĩnh viễn | Win-back campaign hoặc chấp nhận mất — chi phí tái kích hoạt cao |
| **Ngoại lệ** (DBSCAN) | — | — | — | Không thuộc vùng mật độ nào — hành vi bất thường | Phân tích riêng lẻ, phát hiện gian lận hoặc khách hàng đặc biệt |

> DBSCAN bổ sung nhãn **Ngoại lệ** (nhãn -1) cho khách hàng nhiễu.

---

## Đánh giá

| Chỉ số | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| **Silhouette Score** | Độ gắn kết nội cụm / tách biệt liên cụm | Càng gần 1 càng tốt |
| **Davies-Bouldin Index** | Tỉ lệ phân tán trong cụm / khoảng cách giữa cụm | Càng nhỏ càng tốt |

---

## Kết quả thực nghiệm — Mall Customers

### K-Means (k=5)

Chọn k=5 qua Elbow Method + Silhouette Score trên đặc trưng Annual Income × Spending Score.

| Cụm | Income TB (k$) | Score TB | Số KH | Phân khúc |
|-----|---------------|----------|-------|-----------|
| 0 | ~55 | ~50 | 80 | **Khách hàng Ổn định** |
| 1 | ~26 | ~79 | 22 | **Khách hàng Trải nghiệm** |
| 2 | ~26 | ~21 | 23 | **Khách hàng Phổ thông** |
| 3 | ~87 | ~82 | 39 | **Khách hàng Cao cấp** |
| 4 | ~88 | ~18 | 36 | **Khách hàng Thận trọng** |

> Silhouette ≈ **0.5532** · Davies-Bouldin ≈ **0.5711**

### Agglomerative Hierarchical (Ward, n=5)

Dendrogram gợi ý ngưỡng cắt tại khoảng cách ≈ 120 → 5 cụm. Áp dụng **cùng tiêu chí đặt tên** theo tọa độ tâm cụm.

| Phân khúc | Đặc điểm |
|-----------|----------|
| **Khách hàng Cao cấp** | Income cao, Score cao |
| **Khách hàng Thận trọng** | Income cao, Score thấp |
| **Khách hàng Trải nghiệm** | Income thấp, Score cao |
| **Khách hàng Phổ thông** | Income thấp, Score thấp |
| **Khách hàng Ổn định** | Income & Score trung bình |

> Silhouette ≈ **0.5530** · Davies-Bouldin ≈ **0.5782**

### DBSCAN (eps=10, min_samples=3)

`eps=10` xác định từ điểm khuỷu tay trên k-distance graph (k=3).

| Phân khúc | Đặc điểm | Số KH |
|-----------|----------|-------|
| **Khách hàng Ổn định** | Income & Score trung bình | 126 |
| **Khách hàng Trải nghiệm** | Income thấp, Score rất cao | 3 |
| **Khách hàng Cao cấp** | Income cao, Score cao | 33 |
| **Khách hàng Thận trọng** | Income cao, Score thấp | 28 |
| **Điểm ngoại lai** | Không thuộc cụm nào (nhãn -1) | 10 |

> DBSCAN tìm được **4 cụm** thực sự (không phân tách được Phổ thông riêng), 10 điểm nhiễu.

### Tổng hợp so sánh — Mall Customers

| Thuật toán | Số cụm | Silhouette ↑ | Davies-Bouldin ↓ | Nhiễu |
|------------|--------|-------------|-----------------|-------|
| **K-Means** | 5 | 0.5532 | 0.5711 | 0 |
| **Agglomerative** | 5 | 0.5530 | 0.5782 | 0 |
| **DBSCAN** | 4 | 0.3951 | 0.6016 | 10 |

---

## Kết quả thực nghiệm — Online Retail

Notebook: `online_retail_1_classification.ipynb`

### Phương pháp chuẩn hóa

| Biến | Cách tạo | Dùng cho |
|------|----------|----------|
| `X` | Log1p + StandardScaler | EDA visualization |
| `X_mm` | MinMaxScaler (RFM gốc) | K-Means, Agglomerative |
| `X_std` | StandardScaler (RFM gốc) | DBSCAN |

### Cấu hình tham số

| Thuật toán | Tham số | Giá trị | Cách chọn |
|------------|---------|---------|-----------|
| **K-Means** | `n_clusters` | tự động | Elbow — đạo hàm bậc 2 SSE (k=2..10) |
| | `init` | `k-means++` | Khởi tạo tâm cụm thông minh, giảm nguy cơ hội tụ cục bộ |
| | `n_init` | 10 | Chạy 10 lần, giữ kết quả SSE tốt nhất |
| | `random_state` | 42 | Cố định seed để tái tạo kết quả |
| **Agglomerative** | `n_clusters` | tự động | Elbow — đạo hàm bậc 2 SSE (k=2..10) |
| | `linkage` | `ward` | Tối thiểu hóa tăng phương sai khi gộp cụm |
| | `metric` | `euclidean` | Khoảng cách Euclidean |
| **DBSCAN** | `eps` | 0.3 | K-distance graph (k=5) + thực nghiệm theo bài báo |
| | `min_samples` | 5 | Số điểm tối thiểu để tạo core point |

### K-Means

K tối ưu chọn bằng **Elbow Method** (điểm gãy SSE — đạo hàm bậc 2 lớn nhất) trên `X_mm`.

| Cụm | Recency (ngày) | Frequency | Monetary (£) | Phân khúc |
|-----|---------------|-----------|-------------|-----------|
| Khách VIP | thấp nhất | cao nhất | cao nhất | Mua gần đây, thường xuyên, chi tiêu lớn |
| Khách tiềm năng | thấp | trung bình | trung bình | Còn hoạt động, chưa đủ gắn bó |
| Khách rời bỏ | cao nhất | thấp nhất | thấp nhất | Nguy cơ mất vĩnh viễn |

**Tiêu chí đặt tên:** Score tổng hợp = −norm(R) + norm(F) + norm(M). Cụm có score cao nhất → Khách VIP.

### Agglomerative Hierarchical (Ward)

K tối ưu chọn bằng **Elbow Method trên SSE** tính từ AgglomerativeClustering với `X_mm` (k=2..10).
Dendrogram (mẫu 300 KH) vẽ trên `X_mm`, ngưỡng cắt tự động = 60% khoảng cách merge lớn nhất.

### DBSCAN

- Dữ liệu đầu vào: `X_std` (StandardScaler trên RFM gốc)
- `eps = 0.3` (theo thực nghiệm bài báo tham khảo + k-distance graph xác nhận)
- `min_samples = 5`
- Grid search eps = 0.1 → 1.0 để trực quan hóa số cụm và noise theo eps
- Kết quả: **2 cụm** (`Khách hoạt động` và `Khách ít hoạt động`) + 107 điểm ngoại lệ

### Tổng hợp so sánh — Online Retail

| Thuật toán | Chuẩn hóa | Số cụm | Silhouette ↑ | Davies-Bouldin ↓ | Nhiễu |
|------------|-----------|--------|-------------|-----------------|-------|
| **K-Means** | MinMaxScaler | 3 | 0.6448 | 0.5051 | 0 |
| **Agglomerative** | MinMaxScaler | 3 | 0.5537 | 0.5905 | 0 |
| **DBSCAN** | StandardScaler | 2 | 0.6447 | 2.5281 | 107 |

> Phương pháp theo: *John et al., "An Exploration of Clustering Algorithms for Customer Segmentation in the UK Retail Market", Analytics 2023*

---

## So sánh thuật toán

| Thuật toán | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| **K-Means** | Nhanh, dễ triển khai, cụm có ý nghĩa kinh doanh rõ | Cần biết trước K, nhạy cảm với outlier |
| **Agglomerative** | Không cần K trước, dendrogram trực quan | Chậm O(n³), tốn bộ nhớ |
| **DBSCAN** | Phát hiện noise, không cần K, tốt với cụm hình dạng bất kỳ | Khó chọn eps/min_samples, kém với mật độ không đồng đều |

---

## Cấu trúc notebook

```
Mall_Customer_segmentation.ipynb          ← Mall Customers (from scratch)
├── 1. Import thư viện
├── 2. Khám phá dữ liệu (EDA)
├── 3. Hàm tiện ích (euclidean_distance, pairwise_distances, evaluate, plot_clusters)
├── K-Means
│   ├── Class KMeans (from scratch)
│   ├── Elbow Method & Silhouette (k=2..10)
│   ├── Kết quả k=5 + vẽ tâm cụm
│   └── Bảng phân khúc + tiêu chí đặt tên
├── Agglomerative Hierarchical
│   ├── Class Agglomerative (from scratch, Ward linkage)
│   ├── Kết quả n=5
│   ├── Dendrogram
│   └── Bảng phân khúc + tiêu chí đặt tên
├── DBSCAN
│   ├── Class DBSCAN (from scratch)
│   ├── k-distance graph (chọn eps=10)
│   ├── Kết quả eps=10, min_samples=3
│   └── Bảng phân khúc + tiêu chí đặt tên
├── So sánh 3 thuật toán + Kết luận
└── Biểu đồ 2D từng cặp đặc trưng

online_retail_1_classification.ipynb      ← Online Retail (sklearn, MinMaxScaler)
├── 1. Nhập thư viện
├── 2. Tải & khám phá dữ liệu thô (Online Retail.xlsx — 541,909 dòng)
├── 3. Tiền xử lý + xây dựng đặc trưng RFM (4,338 khách hàng)
├── 4. Chuẩn hóa dữ liệu
│   ├── X    — Log1p + StandardScaler (cho EDA)
│   ├── X_mm — MinMaxScaler (cho K-Means & Agglomerative)
│   └── X_std — StandardScaler (cho DBSCAN)
├── 5. Trực quan hóa EDA
│   ├── 5.1 Phân phối RFM (trước vs sau xử lý)
│   ├── 5.2 Boxplot phát hiện ngoại lệ
│   ├── 5.3 Scatter plot các cặp RFM
│   ├── 5.4 Ma trận tương quan
│   ├── 5.5 Top 10 quốc gia theo doanh thu
│   └── 5.6 Doanh thu theo tháng
├── 6. Hàm đánh giá (auto_name_clusters, plot_elbow_silhouette, plot_silhouette, plot_rfm_clusters)
├── 7. K-Means
│   ├── Elbow Method (SSE) trên X_mm (k=2..10)
│   ├── Chọn K tự động bằng đạo hàm bậc 2 SSE
│   ├── Silhouette plot + PCA 2D
│   └── Bảng phân khúc + chiến lược kinh doanh
├── 8. Agglomerative Hierarchical (Ward)
│   ├── Dendrogram (mẫu 300 KH, X_mm)
│   ├── Chọn K tự động bằng đạo hàm bậc 2 SSE
│   └── Silhouette plot + phân tích RFM
├── 9. DBSCAN
│   ├── K-Distance Graph (X_std, eps gợi ý = 0.3)
│   ├── Grid search eps (0.1 → 1.0) trên X_std
│   ├── Huấn luyện với eps=0.3, min_samples=5
│   └── Silhouette plot + phân tích RFM
├── 10. So sánh 3 thuật toán (bảng Silhouette + Davies-Bouldin)
├── 11. Phân tích phân khúc chi tiết (K-Means, bảng lý do + chiến lược)
├── 12. Kết luận
└── 13. Biểu đồ 2D từng cặp RFM (K-Means / Agglomerative / DBSCAN)
```

---

## Yêu cầu

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
openpyxl
```

Cài đặt:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy openpyxl
```

---

## Chạy notebook

```bash
# Mall Customers (from scratch)
jupyter notebook Mall_Customer_segmentation.ipynb

# Online Retail (sklearn)
jupyter notebook online_retail_1_classification.ipynb
```
