# Luong Attention với LSTM (Dot Product Score)

## 1. Kiến trúc tổng quan

Luong Attention là một cơ chế attention được sử dụng phổ biến trong các mô hình Seq2Seq, đặc biệt với LSTM. Luong attention cho phép decoder "chú ý" đến các phần khác nhau của input sequence khi sinh ra output.

Kiến trúc gồm 2 phần chính:

- **Encoder**: LSTM nhận vào chuỗi đầu vào và sinh ra các hidden state $( h_1, h_2, ..., h_T $).
- **Decoder**: LSTM sinh ra chuỗi đầu ra, tại mỗi bước sẽ tính attention dựa trên các hidden state của encoder.

## 2. Quá trình tính Attention (Luong Attention)

### a. Tính score (Dot Product)

Tại mỗi bước $( i $) của decoder, ta tính điểm attention giữa hidden state của decoder $( s_i $) và từng hidden state của encoder $( h_j $):

$$
e_{ij} = s_i^T h_j
$$

### b. Tính trọng số attention (Softmax)

Các score $e_{ij}$ được chuẩn hóa qua hàm softmax để tạo thành trọng số attention:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}
$$

### c. Tính context vector

Context vector $( c_i $) tại bước $( i $) là tổng trọng số của các hidden state encoder:

$$
c_i = \sum_{j=1}^{T} \alpha_{ij} h_j
$$

### d. Sinh output

Context vector $( c_i $) sẽ được kết hợp với hidden state $( s_i $) của decoder để sinh ra output tiếp theo.

## 3. Ưu điểm

- Giúp mô hình "chú ý" tốt hơn tới các phần thông tin quan trọng trong input sequence.
- Cải thiện hiệu quả dịch máy, tổng hợp văn bản, v.v...
