import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

sujitra_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Sujitra\frame_0000.jpg")
sujitra_encoding = face_recognition.face_encodings(sujitra_image)[0]

sarawut_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Sarawut\frame_0000.jpg")
sarawut_encoding = face_recognition.face_encodings(sarawut_image)[0]

kittamat_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Kittamat\frame_0000.jpg")
kittamat_encoding = face_recognition.face_encodings(kittamat_image)[0]

chokun_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Chokun\frame_0000.jpg")
chokun_encoding = face_recognition.face_encodings(chokun_image)[0]

aomsin_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Aomsin\frame_0000.jpg")
aomsin_encoding = face_recognition.face_encodings(aomsin_image)[0]

nort_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Northee\frame_0000.jpg")
nort_encoding = face_recognition.face_encodings(nort_image)[0]

natthawut_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Natthawut\frame_0000.jpg")
natthawut_encoding = face_recognition.face_encodings(natthawut_image)[0]

sutthikan_image = face_recognition.load_image_file(r"C:\CPE\PythonOpencvtest\FaceRecognitionProject\faces_dataset\Sutthikan\frame_0000.jpg")
sutthikan_encoding = face_recognition.face_encodings(sutthikan_image)[0]

know_face_encoding = [
    sujitra_encoding,
    sarawut_encoding,
    kittamat_encoding,
    chokun_encoding,
    aomsin_encoding,
    nort_encoding,
    natthawut_encoding,
    sutthikan_encoding
]

know_face_names = [
    "Sujitra", 
    "Sarawut", 
    "Kittamat", 
    "Chokun",
    "Aomsin", 
    "Nort", 
    "Natthawut", 
    "Sutthikan"
]

students = know_face_names.copy()

face_location = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)
        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(know_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            
            if matches[best_match_index]:
                name = know_face_names[best_match_index]
            
            face_names.append(name)
            if name in know_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()