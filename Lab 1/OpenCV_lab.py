from pathlib import Path
import matplotlib.pyplot as plt
import cv2

folder = Path(__file__).resolve().parent
image = cv2.imread(str(folder / "people.jpg"))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 100)

face_class = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_class.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
face_image2 = image.copy()

for x, y, width, height in faces:
    cv2.rectangle(face_image2,(x, y),(x + width, y + height),(0, 0, 255),2,)

print("Total Number of Faces detected:", len(faces))

fig, graph = plt.subplots(1, 3, figsize=(15, 5))
graph[0].imshow(gray, cmap="gray")
graph[0].set_title("Grayscale")
graph[0].axis("off")
graph[1].imshow(edge, cmap="gray")
graph[1].set_title("Edge Detection")
graph[1].axis("off")
graph[2].imshow(cv2.cvtColor(face_image2, cv2.COLOR_BGR2RGB))
graph[2].set_title("Face Detection")
graph[2].axis("off")

plt.show()