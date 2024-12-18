import cv2
import numpy as np
import imutils
from PIL import Image, ImageTk
from tkinter.ttk import Label
import tkinter as tk

def scan_image():
    try:
        input_path = 'mydoc.jpg'
        img = cv2.imread(input_path)

        if img is None:
            print("Error: Unable to load the image.")
            return

        # Resize image to fit within screen dimensions
        screen_w, screen_h = 1366, 768
        height, width = img.shape[:2]
        scale_ratio = min(screen_w / width, screen_h / height)
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        resized_img = cv2.resize(img, (new_width, new_height))

        # Preprocess image
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 30, 150)

        # Display intermediate processing steps
        cv2.imshow("Gray Image", gray_img)
        cv2.imshow("Blurred Image", blurred_img)
        cv2.imshow("Edge Detection", edges)
        cv2.waitKey(1)

        # Find and sort contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(approx) == 4:  # Only process quadrilaterals
                cv2.drawContours(resized_img, [approx], -1, (0, 0, 255), 3)
                cv2.imshow("Contour Detected", resized_img)
                cv2.waitKey(1)

                # Organize corner points
                approx = approx.reshape(4, 2)
                ordered_corners = np.zeros((4, 2), dtype="float32")
                sum_points = np.sum(approx, axis=1)
                diff_points = np.diff(approx, axis=1)
                ordered_corners[0] = approx[np.argmin(sum_points)]  # Top-left
                ordered_corners[2] = approx[np.argmax(sum_points)]  # Bottom-right
                ordered_corners[1] = approx[np.argmin(diff_points)]  # Top-right
                ordered_corners[3] = approx[np.argmax(diff_points)]  # Bottom-left

                # Compute new dimensions
                (tl, tr, br, bl) = ordered_corners
                width1 = np.linalg.norm(br - bl)
                width2 = np.linalg.norm(tr - tl)
                max_width = max(int(width1), int(width2))

                height1 = np.linalg.norm(tr - br)
                height2 = np.linalg.norm(tl - bl)
                max_height = max(int(height1), int(height2))

                # Map to new rectangle
                destination = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype="float32")

                # Apply perspective transformation
                transformation_matrix = cv2.getPerspectiveTransform(ordered_corners, destination)
                warped_img = cv2.warpPerspective(resized_img, transformation_matrix, (max_width, max_height))

                # Convert to binary
                warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
                _, binary_output = cv2.threshold(warped_gray, 200, 255, cv2.THRESH_BINARY)

                # Show results
                cv2.imshow("Warped Image", warped_img)
                cv2.imshow("Binary Output", binary_output)
                cv2.waitKey(0)
                break  # Stop after processing the first valid contour

    except Exception as err:
        print(f"An unexpected error occurred: {err}")

# GUI Setup
root = tk.Tk()
root.title("Image Scanner Tool")
root.geometry("1322x743")

# Background image for GUI
background = Image.open("wall.jpg").resize((1322, 743))
bg_image = ImageTk.PhotoImage(background)

background_label = Label(root, image=bg_image)
background_label.grid(column=0, row=0)

# Scan Button
scan_btn = tk.Button(root, text="Start Scan", font=("Times New Roman", 20), command=scan_image)
scan_btn.grid(column=0, row=0)

root.mainloop()