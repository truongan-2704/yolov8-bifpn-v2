from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random

# Tải mô hình YOLOv8 đã huấn luyện (có thể là pretrained hoặc mô hình tùy chỉnh của bạn)
model = YOLO('yolov8-bifpn.pt')  # Thay thế bằng mô hình của bạn (ví dụ: 'best.pt')

# Đọc ảnh đầu vào
image_path = 'img_1.png'  # Thay thế với đường dẫn ảnh của bạn
image = cv2.imread(image_path)

# Dự đoán đối tượng trong ảnh
results = model(image)

# Lọc các đối tượng là "car" (class 'car' có thể là class 2 trong COCO dataset)
car_class_index = 1  # Class ID của xe trong COCO dataset
car_boxes = [box for box in results[0].boxes if int(box.cls[0]) == car_class_index]

# Đếm số lượng xe ô tô
num_cars = len(car_boxes)
print(f"Số lượng xe ô tô trong ảnh: {num_cars}")

# Hàm tạo màu ngẫu nhiên
def random_color():
    return [random.randint(0, 255) for _ in range(3)]  # Màu ngẫu nhiên trong dải 0-255 cho mỗi kênh (BGR)

# Vẽ bounding boxes và hiển thị số thứ tự, độ chính xác
for i, box in enumerate(car_boxes):
    x1, y1, x2, y2 = box.xyxy[0]  # Lấy tọa độ bounding box
    confidence = box.conf[0]  # Lấy độ chính xác (confidence score) của đối tượng

    # Tạo màu ngẫu nhiên cho mỗi xe
    color = random_color()

    # Vẽ hình chữ nhật bao quanh xe
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)  # Màu ngẫu nhiên và độ dày 3

    # Hiển thị số thứ tự và độ chính xác trên ảnh
    label = f"Car {i + 1}: {confidence:.2f}"  # Tạo nhãn với số thứ tự và độ chính xác
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (int(x1), int(y1) - 10), font, 0.6, (0, 255, 0), 2)  # Viết nhãn lên ảnh

# Hiển thị tổng số lượng xe trên ảnh
summary_label = f"Total Cars: {num_cars}"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, summary_label, (10, 50), font, 1.0, (255, 255, 0), 2)  # Hiển thị nhãn tổng số xe ở góc trên bên trái

# Hiển thị ảnh với matplotlib (vì ảnh đang ở dạng BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB để hiển thị đúng màu sắc
plt.imshow(image_rgb)
plt.axis('off')  # Ẩn trục tọa độ
plt.show()
