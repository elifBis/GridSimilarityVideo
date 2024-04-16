import cv2
import queue
import functions
import time

video_path = "C://Users//elifb//Downloads//simtask_video.mp4"

def process_video(video_path, scale_percent=50):
    video = cv2.VideoCapture(video_path)
    buffer = queue.Queue()
    
    # İlk kareyi base image olarak belirle
    ret, base_frame = video.read()
    base_frame = cv2.resize(base_frame, (0, 0), fx=scale_percent/100, fy=scale_percent/100)
    base_frame_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    cropped_base_frame = base_frame_gray[380:500, 170:270]

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))
        buffer.put(resized_frame)
        
        cropped_image = resized_frame[380:500, 170:270]
        image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        similarity_score = functions.grid_similarity(cropped_base_frame, image_gray, 5)
        
        cv2.rectangle(resized_frame, (170, 380), (270, 500), (0, 255, 0), 3)
        cv2.imshow('Frame', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(f"Benzerlik Ölçüsü: {similarity_score:.3f}")

        time.sleep(0.025)

    video.release()
    cv2.destroyAllWindows()

process_video(video_path)
