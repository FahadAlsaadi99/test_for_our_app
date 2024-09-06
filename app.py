import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# تحميل النموذج المدرب مسبقاً من YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# تحميل نماذج اكتشاف الوجه والعمر
faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"
ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# تحميل الشبكات
ageNet = cv2.dnn.readNet(ageModel, ageProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

padding = 20

def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

def detect_age(frame, bbox):
    face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                 max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    return ageList[agePreds[0].argmax()]

def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

def calculate_similarity(img1, img2):
    img1_resized = resize_image(img1)
    img2_resized = resize_image(img2)
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def compare_images(dict_in, dict_out):
    images_in = dict_in.get('child', [])
    images_out = dict_out.get('child', [])
    num_images_in = len(images_in)
    num_images_out = len(images_out)
    min_images = min(num_images_in, num_images_out)
    similarity_scores = []
    for i in range(min_images):
        img_in = images_in[i]
        img_out = images_out[i]
        similarity = calculate_similarity(img_in, img_out)
        similarity_scores.append(similarity)
        print(f"تشابه بين الصورة {i+1} في dict_in و الصورة {i+1} في dict_out: {similarity:.2f}")
    if num_images_in > num_images_out:
        for i in range(num_images_out, num_images_in):
            print(f"لا توجد صورة مقابلة للصورة {i+1} في dict_in في dict_out.")
    elif num_images_out > num_images_in:
        for i in range(num_images_in, num_images_out):
            print(f"لا توجد صورة مقابلة للصورة {i+1} في dict_out في dict_in.")
    threshold = 0.8
    found_similar = any(score > threshold for score in similarity_scores[:3])
    if found_similar:
        print("الطفل موجود مع أحد الوالدين.")
    else:
        print("تحذير: قد يكون الطفل مختطفاً.")

# واجهة المستخدم Streamlit
st.title("كشف الأطفال والمقارنة بين الصور")

uploaded_file1 = st.file_uploader("اختر الصورة الأولى", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("اختر الصورة الثانية", type=["jpg", "jpeg", "png"])

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)
    
    image_cv1 = np.array(image1)
    image_cv2 = np.array(image2)
    
    results1 = model(image1)
    predictions1 = results1.pandas().xyxy[0]
    
    results2 = model(image2)
    predictions2 = results2.pandas().xyxy[0]
    
    cropped_objects_dict_in = {}
    cropped_objects_dict_out = {}

    # معالجة الصورة الأولى
    found_child1 = False
    child_bbox1 = None
    for idx, row in predictions1.iterrows():
        x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
        if results1.names[int(cls)] == "person":
            face_bboxes = getFaceBox(faceNet, image_cv1)
            for bbox in face_bboxes:
                age = detect_age(image_cv1, bbox)
                if age in ['(0-2)', '(4-6)', '(8-12)']:
                    found_child1 = True
                    child_bbox1 = (x1, y1, x2, y2)
                    break
            if found_child1:
                break
    
    if found_child1:
        all_objects1 = []
        for idx, row in predictions1.iterrows():
            x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
            label = f"{results1.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image_cv1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_cv1, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cropped_object = image_cv1[int(y1):int(y2), int(x1):int(x2)]
            if (x1, y1, x2, y2) == child_bbox1:
                all_objects1.append(cropped_object)
            else:
                all_objects1.append(cropped_object)
        cropped_objects_dict_in["child"] = all_objects1

    # معالجة الصورة الثانية
    found_child2 = False
    child_bbox2 = None
    for idx, row in predictions2.iterrows():
        x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
        if results2.names[int(cls)] == "person":
            face_bboxes = getFaceBox(faceNet, image_cv2)
            for bbox in face_bboxes:
                age = detect_age(image_cv2, bbox)
                if age in ['(0-2)', '(4-6)', '(8-12)']:
                    found_child2 = True
                    child_bbox2 = (x1, y1, x2, y2)
                    break
            if found_child2:
                break

    if found_child2:
        all_objects2 = []
        for idx, row in predictions2.iterrows():
            x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
            label = f"{results2.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_cv2, label, (int(x1), int(y1) -
