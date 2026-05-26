# Phân Khúc Khách Hàng — Customer Segmentation from Scratch

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
└── So sánh 3 thuật toán + Kết luận
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

## Yêu cầu

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

Cài đặt:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## Chạy notebook

```bash
jupyter notebook customer-segmentation-scratch.ipynb
```
