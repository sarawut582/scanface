import os
import cv2

# พาธของฐานข้อมูลใบหน้า
face_database_path = 'C:/CPE/PythonOpencvtest/FaceRecognitionProject/faces_dataset/Sujitra'

# ฟังก์ชันตรวจจับใบหน้า
def detect_face(img):
    # ใช้ cascade classifier ของ OpenCV สำหรับตรวจจับใบหน้า
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return img
    

# ลิสต์ไฟล์ทั้งหมดในโฟลเดอร์
for img_filename in os.listdir(face_database_path):
    img_path = os.path.join(face_database_path, img_filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"ไม่สามารถโหลดภาพ: {img_path}")
        continue

    # แสดงข้อความกำลังประมวลผลภาพ
    print(f"กำลังประมวลผลภาพ: {img_filename}")
    
    # ตรวจจับใบหน้าในภาพ
    img_with_face = detect_face(img)
    
    # ลดขนาดรูปภาพก่อนแสดงผล
    resized_img = cv2.resize(img_with_face, (640, 480))  # ปรับขนาดเป็น 640x480

    # แสดงภาพที่มีการตรวจจับใบหน้า
    cv2.imshow(f"Detected Face in {img_filename}", resized_img)

    cv2.waitKey(0)

# ปิดหน้าต่างแสดงผล
cv2.destroyAllWindows()
