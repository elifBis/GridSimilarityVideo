import cv2
import threading
import queue
import time
import functions

video_path = "C://Users//elifb//Downloads//simtask_video.mp4"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)


buffer = queue.Queue()
sim_event = threading.Event()

# İlk karenin işleneceği bayrak
first_frame_processed = False

# İlk karenin işlenmesi
def process_first_frame(frame):
    global first_frame_processed
    base_frame = frame
    if base_frame is None:
        print("base_frame yüklenemedi.")
        return None
    base_frame_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    cropped_base_frame_gray = base_frame_gray[380:500, 170:270]
    # cropped_base_frame_gray = base_frame_gray[300:450, 650:750]
    # cropped_base_frame_gray = base_frame_gray[300:450, 650:750]

    first_frame_processed = True
    return cropped_base_frame_gray

def process_video():
    while True:
        frame = buffer.get()
        if not first_frame_processed:
            cropped_base_frame_gray = process_first_frame(frame)
            if cropped_base_frame_gray is None:
                continue

        cropped_image = frame[380:500, 170:270]
        # cropped_image = frame[300:450, 650:750]
        # cropped_image = frame[300:450, 650:750]

        image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # print("cropped_base_frame_gray",cropped_base_frame_gray.shape)
        # print("image_gray.", image_gray.shape)


        similarity_score = functions.grid_similarity(cropped_base_frame_gray, image_gray, 5)

        print(similarity_score)

        if similarity_score < 13:
            sim_event.set()
        else:
            sim_event.clear()

def Mesaj():
    while True:
        if sim_event.is_set():
            print("Nesne Kaldirildi")
        time.sleep(0.1)

video = cv2.VideoCapture(video_path)

thread1 = threading.Thread(target=process_video)
thread1.start()
thread2 = threading.Thread(target=Mesaj)
thread2.start()

while True:
    ret, frame = video.read()
    if not ret:
        break

    width = int(frame.shape[1] * 50 / 100)
    height = int(frame.shape[0] * 50 / 100)
    resized_frame = cv2.resize(frame, (width, height))
    cv2.rectangle(resized_frame, (170, 380), (270, 500), (0, 255, 0), 3)
    # # cv2.rectangle(resized_frame, (650, 300), (750, 450), (0, 255, 0), 3)
    # cv2.rectangle(resized_frame, (600, 170), (720, 330), (0, 255, 0), 3)


    buffer.put(resized_frame)
    cv2.imshow('Frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
