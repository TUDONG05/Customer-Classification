# Phân Khúc Khách Hàng — Customer Segmentation 

Triển khai thủ công ba thuật toán phân cụm  **K-Means**, **DBSCAN**, **Agglomerative Hierarchical** — trên bộ dữ liệu Mall Customers.

---

## Bộ dữ liệu

**Mall_Customers.csv** — 200 khách hàng, 5 thuộc tính:

| Cột | Mô tả |
|-----|-------|
| `CustomerID` | Mã khách hàng |
| `Gender` | Giới tính |
| `Age` | Tuổi |
| `Annual Income (k$)` | Thu nhập hàng năm (nghìn USD) |
| `Spending Score (1-100)` | Điểm chi tiêu tại trung tâm thương mại |

Các thuật toán phân cụm sử dụng hai đặc trưng `Annual Income (k$)` và `Spending Score (1-100)`.

---

## Thuật toán

### K-Means
- Khởi tạo K tâm cụm ngẫu nhiên
- Gán mỗi điểm vào tâm gần nhất (khoảng cách Euclidean)
- Cập nhật tâm = trung bình các điểm trong cụm
- Lặp đến hội tụ (tâm không dịch chuyển quá `tol`)
- Chọn k tối ưu qua **Elbow Method** (SSE) và **Silhouette Score** → k=5

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
- Chọn `eps` qua **k-distance graph** (điểm khuỷu tại eps=10)

---

## Cấu trúc notebook

```
customer-segmentation-scratch.ipynb
├── 1. Import thư viện
├── 2. Đọc dữ liệu
├── 3. Hàm tiện ích (euclidean_distance, pairwise_distances, evaluate, plot_clusters)
├── K-Means
│   ├── Class KMeans (from scratch)
│   ├── Elbow Method & Silhouette (k=2..10)
│   └── Kết quả k=5 + vẽ tâm cụm
├── Agglomerative Hierarchical
│   ├── Class Agglomerative (from scratch, Ward linkage)
│   ├── Kết quả n=5
│   └── Dendrogram
├── DBSCAN
│   ├── Class DBSCAN (from scratch)
│   ├── k-distance graph (chọn eps)
│   └── Kết quả eps=10, min_samples=3
├── So sánh 3 thuật toán + Kết luận
└── Biểu đồ 3D xoay được (Plotly) → xuất 3d_kmeans.html, 3d_agglomerative.html, 3d_dbscan.html
```

---

## Đánh giá

Hai chỉ số dùng để so sánh (tính bằng `sklearn.metrics`):

| Chỉ số | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| **Silhouette Score** | Độ gắn kết nội cụm / tách biệt liên cụm | Càng gần 1 càng tốt |
| **Davies-Bouldin Index** | Tỉ lệ phân tán trong cụm / khoảng cách giữa cụm | Càng nhỏ càng tốt |

Kết quả tốt nhất trên Mall Customers: **K-Means k=5** (Annual Income × Spending Score).

---

## So sánh thuật toán

| Thuật toán | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| **K-Means** | Nhanh, dễ triển khai | Cần biết trước K, nhạy cảm với outlier |
| **Agglomerative** | Không cần K trước, dendrogram trực quan | Chậm O(n³), tốn bộ nhớ |
| **DBSCAN** | Phát hiện noise, không cần K | Khó chọn eps/min_samples, kém với mật độ không đồng đều |

---

## Bộ dữ liệu Online Retail II

**online_retail_II.xlsx** — giao dịch thực tế của cửa hàng bán lẻ trực tuyến tại Anh, giai đoạn 01/12/2009 – 09/12/2011.

| Cột | Mô tả |
|-----|-------|
| `Invoice` | Mã hóa đơn (bắt đầu bằng "C" = hóa đơn hủy) |
| `StockCode` | Mã sản phẩm |
| `Description` | Tên sản phẩm |
| `Quantity` | Số lượng |
| `InvoiceDate` | Ngày giờ hóa đơn |
| `Price` | Đơn giá (£) |
| `Customer ID` | Mã khách hàng (22.8% null) |
| `Country` | Quốc gia |

**Sau tiền xử lý:** 805,549 dòng — 5,878 khách hàng duy nhất.

### Đặc trưng RFM

Mỗi khách hàng được biểu diễn bằng 3 chỉ số RFM thay vì các cột giao dịch thô:

| Đặc trưng | Ý nghĩa | Thống kê |
|-----------|---------|---------|
| **Recency (R)** | Số ngày kể từ lần mua cuối | TB: 201 ngày · max: 739 ngày |
| **Frequency (F)** | Số hóa đơn khác nhau | TB: 6.3 · max: 398 |
| **Monetary (M)** | Tổng doanh thu (£) | TB: £3,019 · max: £608,822 |

Quy trình chuẩn hóa: **Log transform** (giảm skew) → **StandardScaler**.

---

## Kết quả thực nghiệm — Online Retail II

### K-Means (k=4)

Chọn k=4 qua Elbow Method + Silhouette Score.

| Cụm | Recency (ngày) | Frequency | Monetary (£) | Số KH | % KH | Phân khúc |
|-----|---------------|-----------|-------------|-------|------|-----------|
| 0 | 27.4 | 19.3 | 11,014 | 1,188 | 20.2% | **VIP / Trung thành** |
| 1 | 395.9 | 1.4 | 326 | 1,974 | 33.6% | **Không hoạt động** |
| 2 | 227.9 | 5.1 | 2,002 | 1,465 | 24.9% | **Tiềm năng** |
| 3 | 28.4 | 3.0 | 865 | 1,251 | 21.3% | **Khách mới** |

> Silhouette = **0.3653** · Davies-Bouldin = **0.9303**

### Agglomerative Hierarchical (Ward, k=4)

Dendrogram (mẫu 300 KH) gợi ý ngưỡng cắt tại khoảng cách ≈ 6 → 4 cụm.

| Cụm | Số KH |
|-----|-------|
| 0 | 1,831 |
| 1 | 1,620 |
| 2 | 1,472 |
| 3 | 955 |

> Silhouette = **0.3031** · Davies-Bouldin = **0.9584**

### DBSCAN (eps tự động, min_samples=5)

`eps` được chọn qua k-distance graph và grid search — ưu tiên eps lớn nhất còn tạo được ≥ 2 cụm với noise < 20%.

| Chỉ số | Giá trị |
|--------|---------|
| Số cụm | 2 |
| Điểm nhiễu | 21 (0.4%) |
| Silhouette | **0.62** |
| Davies-Bouldin | **0.34** |

> DBSCAN cho Silhouette cao nhất và Davies-Bouldin thấp nhất, nhưng chỉ phân được 2 cụm — hữu ích để phát hiện khách hàng bất thường hơn là phân khúc chi tiết.

### Tổng hợp so sánh (Online Retail II)

| Thuật toán | Số cụm | Silhouette ↑ | Davies-Bouldin ↓ | Nhiễu |
|------------|--------|-------------|-----------------|-------|
| **K-Means** | 4 | 0.3653 | 0.9303 | 0 |
| **Agglomerative** | 4 | 0.3031 | 0.9584 | 0 |
| **DBSCAN** | 2 | **0.6200** | **0.3400** | 21 |

**Nhận xét:** K-Means cho kết quả phân khúc kinh doanh rõ ràng nhất (4 nhóm có ý nghĩa). DBSCAN vượt trội về chỉ số mật độ nhưng chỉ tách được 2 nhóm lớn trên dữ liệu RFM.

---

## Yêu cầu

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
plotly
openpyxl
```

Cài đặt:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly openpyxl
```

---

## Chạy notebook

```bash
# Mall Customers (from scratch)
jupyter notebook customer-segmentation-scratch.ipynb

# Online Retail II (sklearn)
jupyter notebook online_retail_clustering_vi.ipynb
```
