import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from facenet_pytorch import MTCNN
import mediapipe as mp

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_model(onnx_path):
    """โหลด ONNX Model และสร้าง TensorRT Engine"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # ตรวจสอบ Dynamic Shape
    input_tensor = network.get_input(0)
    if input_tensor.shape[0] == -1:
        print("⚠️ Detected Dynamic Batch Size, setting optimization profile...")
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, (1, 3, 160, 160), (4, 3, 160, 160), (8, 3, 160, 160))
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ Failed to serialize TensorRT engine!")
        return None

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)

onnx_path = "C:/CPE/PythonOpencvtest/facenet_fixed.onnx"
trt_engine = load_trt_model(onnx_path)

if trt_engine:
    context = trt_engine.create_execution_context()
    print("✅ TensorRT Engine Loaded Successfully!")
else:
    print("❌ ไม่สามารถโหลด TensorRT engine ได้!")

mtcnn = MTCNN(image_size=240, margin=40, min_face_size=50)  # เพิ่ม min_face_size เพื่อความแม่นยำ
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

face_embeddings = np.load("face_database.npy", allow_pickle=True).item()

def recognize_face_with_trt(face_resized):
    """ ใช้ TensorRT ในการทำ inference สำหรับ FaceNet """
    if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
        # แปลงจาก BGR เป็น RGB และ Normalize ให้มีค่าในช่วง [-1, 1]
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = (face_resized / 255.0 - 0.5) * 2  # Normalize ให้มีค่าในช่วง [-1, 1]
        input_data = np.transpose(face_resized, (2, 0, 1)).astype(np.float32).flatten()
    else:
        print("❌ พบปัญหากับรูปภาพที่ส่งเข้าไป face_resized ต้องมี 3 ช่อง")
        return None

    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)

    output_data = np.empty([512], dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)

    context.execute_v2([int(d_input), int(d_output)])

    cuda.memcpy_dtoh(output_data, d_output)
    return output_data

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

video_path = r"C:\CPE\PythonOpencvtest\video\test11.MOV"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("\nError: ไม่สามารถเปิดไฟล์วิดีโอได้!")
    exit()

# เพิ่มการปรับขนาดของเฟรมสำหรับการประมวลผลที่เร็วขึ้น
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_resized_width = 640  # ลดขนาดเฟรมเพื่อเพิ่มความเร็ว
frame_resized_height = int((frame_resized_width / frame_width) * frame_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ลดขนาดเฟรมเพื่อลดภาระการประมวลผล
    frame_resized = cv2.resize(frame, (frame_resized_width, frame_resized_height))

    # หมุนเฟรมให้เป็นแนวนอน (landscape)
    frame_resized = cv2.transpose(frame_resized)
    frame_resized = cv2.flip(frame_resized, flipCode=1)  # 1: หมุนภาพ 90 องศา (แนวนอน)

    # ตรวจจับใบหน้า
    boxes, _ = mtcnn.detect(frame_resized)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face = frame_resized[y:y + h, x:x + w]
            if face.size > 0:
                face_resized = cv2.resize(face, (160, 160))

                # ตรวจสอบการสร้าง face embedding
                face_embedding = recognize_face_with_trt(face_resized)

                if face_embedding is None:
                    continue  # หากไม่สามารถสร้าง face embedding ได้ ให้ข้ามไป

                name = "Unknown"
                best_similarity = float("-inf")

                # เปรียบเทียบกับฐานข้อมูลใบหน้า
                for db_name, db_embedding in face_embeddings.items():
                    similarity = cosine_similarity(face_embedding, db_embedding)
                    print(f"Comparing with {db_name}, Similarity: {similarity:.2f}")  # Debugging output

                    if similarity > best_similarity:
                        best_similarity = similarity
                        name = db_name

                confidence = best_similarity  # Cosine similarity อยู่ในช่วง -1 ถึง 1

                # หากความแตกต่างมากเกินไปให้แสดงว่าไม่รู้จัก
                if best_similarity < 0.6:
                    name = "Unknown"

                # วาดกรอบใบหน้าตามที่ต้องการ
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame_resized, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # ใช้ MediaPipe เพื่อจับจุดใบหน้า
                results = mp_face_mesh.process(frame_resized)
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        for landmark in landmarks.landmark:
                            h, w, c = frame_resized.shape
                            x_landmark = int(landmark.x * w)
                            y_landmark = int(landmark.y * h)

                            # วาดจุดใบหน้าบนภาพ
                            cv2.circle(frame_resized, (x_landmark, y_landmark), 1, (0, 0, 255), -1)

    # แสดงภาพที่มีการวิเคราะห์ใบหน้า
    cv2.imshow("Face Recognition + Liveness Detection with TensorRT", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
