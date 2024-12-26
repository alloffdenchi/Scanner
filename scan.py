import cv2
import numpy as np
import imutils
from PIL import Image, ImageTk
from tkinter.ttk import Label
import tkinter as tk

def scan2():
    try:
        # Tên file ảnh đầu vào
        input_image = 'mydoc.jpg'

        # Đọc ảnh
        image = cv2.imread(input_image)

        if image is None:
            print("Error: Cannot read input image.")
            return

        # Resize image to fit screen
        screen_width = 1366
        screen_height = 768
        h, w = image.shape[:2]
        scale = min(screen_width / w, screen_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))

        # Chuyển ảnh màu thành ảnh xám
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Làm mờ ảnh xám để xóa noise bằng cách áp dụng bộ lọc Gaussian Blur.
        blur = cv2.blur(gray, (3, 3))

        # Tìm cạnh bằng thuật toán Canny
        edge = cv2.Canny(blur, 30, 150, 3)

        # Hiển thị các bước xử lý ảnh
        cv2.imshow("Gray Image", gray)
        cv2.imshow("Blurred Image", blur)
        cv2.imshow("Edge Detection", edge)
        cv2.waitKey(1)

        # Tìm contours
        cnts = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Sắp xếp theo diện tích giảm dần
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        if not cnts:
            print("No contours found.")
            return

        # Lấy contour lớn nhất
        largest_contour = cnts[0]

        # Xấp xỉ đa giác để lấy hình chữ nhật
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

        if len(approx) != 4:
            print("Contour approximation did not return a quadrilateral.")
            return

        # Vẽ contour lên ảnh gốc
        cv2.drawContours(resized_image, [approx], -1, (0, 255, 0), 3)
        cv2.imshow("Contour Drawing", resized_image)
        cv2.waitKey(1)

        # Sắp xếp các điểm góc
        approx = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Tính tổng và hiệu để xác định vị trí các góc
        s = np.sum(approx, axis=1)
        diff = np.diff(approx, axis=1)

        rect[0] = approx[np.argmin(s)]  # Top-left
        rect[2] = approx[np.argmax(s)]  # Bottom-right
        rect[1] = approx[np.argmin(diff)]  # Top-right
        rect[3] = approx[np.argmax(diff)]  # Bottom-left

        # Tính chiều rộng và chiều cao mới
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)) #kclonnhatgiuatr
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) #kclonnhatgiuatl
        maxHeight = max(int(heightA), int(heightB)) 

        # Mảng tọa độ chứa điểm góc của HCN mới
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Tính mtr M va doi phoi canh anh goc bang M
        M = cv2.getPerspectiveTransform(rect, dst) #diemgocrectsangdst
        warped = cv2.warpPerspective(resized_image, M, (maxWidth, maxHeight)) 

        # Chuyển ảnh warp sang ảnh xám và áp dụng threshold
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY) #den<trang

        # Hiển thị kết quả cuối
        cv2.imshow("Warped Image", warped)
        cv2.imshow("Thresholded Image", warped_thresh)
        cv2.waitKey(1)

    except Exception as e:
        print(f"An error occurred: {e}")

# Giao diện chính
giaodien = tk.Tk()
giaodien.title("Welcome to My App")
giaodien.geometry("1322x743")

# Đưa ảnh vào giao diện
bia = Image.open("wall.jpg")
resize_image = bia.resize((1322, 743))
background_image = ImageTk.PhotoImage(resize_image)

img_label = Label(giaodien, image=background_image)
img_label.grid(column=0, row=0)

# Thêm nút bắt đầu scan
scan_button = tk.Button(giaodien, text="Start Scan", font=("Times New Roman", 20), command=scan2)
scan_button.grid(column=0, row=0)

giaodien.mainloop()
