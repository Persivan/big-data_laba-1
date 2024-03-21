import cv2

# Загрузка изображения
image = cv2.imread("faces.jpg")
cv2.imshow("Original", image)
cv2.waitKey(0)
# преобразуем изображение к оттенкам серого
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# инициализировать распознаватель лиц (каскад Хаара по умолчанию)
face_cascade = cv2.CascadeClassifier("Haarcascade_frontalface_default.xml")

# обнаружение всех лиц на изображении
faces = face_cascade.detectMultiScale(image_gray)
# печатать количество найденных лиц
print(f"{len(faces)} лиц обнаружено на изображении.")

# для всех обнаруженных лиц рисуем синий квадрат
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

# сохраним изображение с обнаруженными лицами
cv2.imwrite("2faces.jpg", image)

#ЧТЕНИЕ С КАМЕРЫ
# создать новый объект камеру
cap = cv2.VideoCapture(0)

while True:
    # чтение изображения с камеры
    _, image = cap.read()
    # преобразование к оттенкам серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # обнаружение лиц на фотографии
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    # для каждого обнаруженного лица нарисовать синий квадрат
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
        cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
