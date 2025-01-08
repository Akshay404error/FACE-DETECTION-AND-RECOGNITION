import cv2
import face_recognition
import numpy as np
import csv
import os

# File to store the dataset
DATASET_FILE = "faces_dataset.csv"


def load_known_faces(dataset_file=DATASET_FILE):
    """
    Load known face encodings and names from the dataset file.
    """
    if not os.path.exists(dataset_file):
        print(f"No dataset found at {dataset_file}. Starting fresh!")
        return [], []

    known_face_encodings = []
    known_face_names = []

    with open(dataset_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            name, encoding_str = row[0], row[1]
            encoding = np.fromstring(encoding_str, sep=' ')
            known_face_names.append(name)
            known_face_encodings.append(encoding)

    return known_face_encodings, known_face_names


def save_face_data(name, encoding, dataset_file=DATASET_FILE):
    """
    Save face encoding and name to the dataset file.
    """
    with open(dataset_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, ' '.join(map(str, encoding))])
    print(f"Saved face data for {name}.")


def capture_face_data():
    """
    Capture face data from the webcam and save it to the dataset file.
    """
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to save a face and 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        cv2.imshow("Capture Face Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            name = input("Enter the name of the person: ")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                save_face_data(name, face_encoding)
            else:
                print("No face detected. Try again.")

        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def detect_and_recognize_faces():
    """
    Detect and recognize faces from the webcam feed using the dataset.
    """
    known_face_encodings, known_face_names = load_known_faces()
    if not known_face_encodings:
        print("No known faces found. Please capture face data first.")
        return

    video_capture = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Detection and Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1: Capture Face Data")
    print("2: Detect and Recognize Faces")
    print("3: Exit")

    while True:
        choice = input("Enter your choice: ")
        if choice == "1":
            capture_face_data()
        elif choice == "2":
            detect_and_recognize_faces()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
