# Tìm Hiểu Về Nền Tảng Machine Learning/Deep Learning + Vector Hóa

Machine Learning (ML) và Deep Learning (DL) là các lĩnh vực cốt lõi trong trí tuệ nhân tạo (AI), tập trung vào việc xây dựng mô hình có khả năng học từ dữ liệu để đưa ra dự đoán hoặc quyết định. Deep Learning là một nhánh của Machine Learning sử dụng mạng nơ-ron nhân tạo với nhiều lớp để xử lý dữ liệu phức tạp như hình ảnh, văn bản hoặc âm thanh. Vector hóa là bước quan trọng để chuyển đổi dữ liệu thô thành dạng số mà máy tính có thể xử lý. Trong tài liệu này, chúng ta sẽ tập trung vào học có giám sát (Supervised Learning), các khái niệm cơ bản của mạng nơ-ron, và quy trình huấn luyện mô hình.

## Supervised Learning - Học Có Giám Sát

### Khái Niệm
Học có giám sát (Supervised Learning) là một phương pháp huấn luyện mô hình machine learning từ dữ liệu đã được gán nhãn đầy đủ. Dữ liệu bao gồm cặp input (đầu vào) và output (đầu ra) tương ứng, nơi output là "nhãn" đúng (ground truth). Mô hình học cách ánh xạ từ input sang output bằng cách tìm ra các mẫu (patterns) trong dữ liệu.

Mục tiêu chính là giảm thiểu sai số giữa dự đoán của mô hình và nhãn thực tế. Phương pháp này thường được sử dụng cho các bài toán phân loại (classification) hoặc hồi quy (regression):
- **Phân loại**: Dự đoán lớp (category), ví dụ: phân loại email là spam hay không spam.
- **Hồi quy**: Dự đoán giá trị liên tục, ví dụ: dự đoán giá nhà dựa trên diện tích và vị trí.

Ưu điểm: Độ chính xác cao nếu dữ liệu nhãn tốt. Nhược điểm: Cần nhiều dữ liệu nhãn, tốn kém để thu thập và gán nhãn.

### Quy Trình Huấn Luyện
Quy trình học có giám sát thường bao gồm các bước sau:

1. **Thu Thập Dữ Liệu Có Nhãn**: Thu thập dữ liệu đầu vào (features) và nhãn tương ứng. Ví dụ: Trong bài toán nhận diện chữ viết tay (OCR - Optical Character Recognition), input là ảnh chứa chữ, output là văn bản đúng (e.g., ảnh chữ "A" → nhãn "A").
   
2. **Chia Dữ Liệu**: Phân chia dữ liệu thành:
   - **Tập Huấn Luyện (Training Set)**: 70-80% dữ liệu, dùng để huấn luyện mô hình.
   - **Tập Kiểm Tra (Validation Set)**: 10-15%, dùng để điều chỉnh siêu tham số (hyperparameters) và tránh overfitting.
   - **Tập Thử Nghiệm (Test Set)**: 10-15%, dùng để đánh giá cuối cùng.

3. **Huấn Luyện Mô Hình Và Dự Đoán**: Sử dụng thuật toán (e.g., mạng nơ-ron) để học từ training set và đưa ra dự đoán trên validation/test set.

4. **Đánh Giá Kết Quả**: Sử dụng các chỉ số đo lường hiệu suất:
   - **Accuracy**: Tỷ lệ dự đoán đúng, với công thức:
     $$
     Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
     $$
     (TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative)
   - **Precision**: Độ chính xác, với công thức:
     $$
     Precision = \frac{TP}{TP + FP}
     $$
   - **Recall**: Độ bao phủ, với công thức:
     $$
     Recall = \frac{TP}{TP + FN}
     $$
   - **F1-Score**: Trung bình điều hòa giữa Precision và Recall:
     $$
     F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
     $$
   - **Mean Squared Error (MSE)**: Cho hồi quy:
     $$
     MSE = \frac{\sum (y_i - \hat{y}_i)^2}{n}
     $$
   - **Confusion Matrix**: Bảng hiển thị TP, FP, TN, FN.

### Ví Dụ
- **OCR Trong Dịch Ảnh**: Input: Ảnh chứa chữ (e.g., ảnh biển số xe). Output: Văn bản đúng (e.g., "ABC-123"). Mô hình học cách trích xuất và nhận diện ký tự.
- **Dự Đoán Bệnh Lý**: Input: Hình ảnh X-quang. Output: Nhãn "có bệnh" hoặc "không bệnh".
- **Dịch Máy**: Input: Câu tiếng Anh. Output: Câu tiếng Việt đúng.

## Vector Hóa Dữ Liệu

Máy tính chỉ xử lý dữ liệu số (numbers), nên cần chuyển đổi dữ liệu thô như text, ảnh, âm thanh thành dạng vector (mảng số). Vector hóa giúp mô hình học được các đặc trưng (features) và quan hệ giữa dữ liệu.

### Khái Niệm Và Phương Pháp
- **One-Hot Encoding**: Dùng cho dữ liệu rời rạc (categorical). Nếu có $k$ lớp, thì mỗi lớp được biểu diễn bởi một vector chiều $k$ với chỉ một phần tử bằng 1, các phần tử còn lại bằng 0.
- **Nhúng (Embedding)**: Biểu diễn dense vector (mảng số thực ngắn gọn) giữ được quan hệ ngữ nghĩa, thường có kích thước nhỏ hơn one-hot(Mỗi vector one-hot có kích thước 10,000 chiều (99.99% là số 0, chỉ 1 giá trị là 1).

- Nhược điểm: vector rất dài, rất sparse (thưa thớt), tốn RAM.). Embedding học từ dữ liệu, nơi các vector gần nhau có nghĩa tương tự.

### Ví Dụ
- **Text Embedding**: 
  - "cat" = $[0.12, -0.23, 0.55, ...]$ (vector 300 chiều).
  - "dog" = $[0.15, -0.20, 0.50, ...]$ (gần "cat" hơn "car").
  - Khoảng cách: Sử dụng cosine similarity:
    $$
    \text{Cosine}(A, B) = \frac{A \cdot B}{|A| \cdot |B|}
    $$
- **Ảnh Trong Bài Toán Dịch Ảnh**: Sử dụng Convolutional Neural Network (CNN) để trích xuất đặc trưng:
  - Input: Ảnh (ma trận pixel).
  - CNN: Áp dụng filter để tạo feature map.
  - Flatten: Chuyển feature map thành vector 1D, ví dụ: $[0.3, 0.7, -0.1, ...]$.
- **Âm Thanh**: Spectrogram $\rightarrow$ Vector qua Mel-Frequency Cepstral Coefficients (MFCC).

## Mạng Nơ-Ron Cơ Bản

Mạng nơ-ron nhân tạo (Neural Network) mô phỏng não bộ, gồm nhiều lớp nơ-ron kết nối.

### Neuron - Thành Phần Cơ Bản
Mỗi neuron nhận input, tính toán:
$$
z = \sum_{i} w_i x_i + b
$$
Sau đó áp dụng hàm kích hoạt (activation function):
$$
a = f(z)
$$

### Activation Function
Một số hàm kích hoạt thường dùng:
- **ReLU (Rectified Linear Unit)**: 
  $$
  f(x) = \max(0, x)
  $$
- **Sigmoid**: 
  $$
  f(x) = \frac{1}{1 + e^{-x}}
  $$
- **Tanh**: 
  $$
  f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  $$
- **Softmax**: 
  $$
  f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
  $$

### Forward Pass
Dữ liệu đi từ input layer qua hidden layers đến output layer:
- Input $\rightarrow$ Multiply weights + bias $\rightarrow$ Activation $\rightarrow$ Next layer.

### Backward Pass (Backpropagation)
- Tính gradient của loss theo từng trọng số bằng chain rule.
- Cập nhật trọng số:
  $$
  w_\text{new} = w_\text{old} - \eta \cdot \text{gradient}
  $$
  ($\eta$: learning rate)

## Loss Function - Hàm Mất Mát

Loss function đo lường độ sai lệch giữa dự đoán và nhãn thực tế. Mô hình tối ưu để minimize loss.

- **Cross-Entropy Loss**: Dùng cho phân loại (categorical).
  $$
  L = -\sum_{i} y_i \log(\hat{y}_i)
  $$
  (với $y$: one-hot vector nhãn thật, $\hat{y}$: xác suất dự đoán)
- **Mean Squared Error (MSE)**: Cho hồi quy:
  $$
  MSE = \frac{\sum (y_i - \hat{y}_i)^2}{n}
  $$
- **Binary Cross-Entropy**: Cho phân loại nhị phân:
  $$
  L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
  $$

## Optimizer - Tối Ưu Hóa

Optimizer cập nhật trọng số dựa trên gradient để minimize loss.

- **SGD (Stochastic Gradient Descent)**:
  $$
  w = w - \eta \cdot \nabla L
  $$
- **Adam (Adaptive Moment Estimation)**:
  Adam sử dụng các biến động (momentum) và learning rate thích nghi cho từng tham số.
  - Tính các giá trị trung bình động $m_t$ (first moment) và $v_t$ (second moment) theo gradient.
  - Cập nhật trọng số:
    $$
    w = w - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
    $$
- **Các Optimizer Khác**: RMSProp, Adagrad.

## Quy Trình Tổng Quát Huấn Luyện Mô Hình

Dưới đây là quy trình tổng quát cho một mô hình Deep Learning trong học có giám sát:

1. **Chuẩn Bị Dữ Liệu**: Thu thập ảnh/text $\rightarrow$ Vector hóa (Embedding cho text, CNN feature extraction cho ảnh, hoặc One-hot).
2. **Xây Dựng Mô Hình**: Khởi tạo mạng nơ-ron với layers, activation functions.
3. **Forward Pass**: Dữ liệu vector qua mạng $\rightarrow$ Dự đoán output.
4. **Tính Loss**: So sánh dự đoán với nhãn thật (e.g., Cross-Entropy cho phân loại).
5. **Backward Pass**: Tính gradient qua backpropagation.
6. **Tối Ưu Hóa**: Cập nhật trọng số bằng optimizer (SGD/Adam). Lặp qua nhiều epoch (e.g., 10-100 epoch), theo dõi loss trên validation set để tránh overfitting (sử dụng regularization như dropout).
7. **Đánh Giá Và Triển Khai**: Test trên test set $\rightarrow$ Deploy mô hình nếu đạt yêu cầu.

