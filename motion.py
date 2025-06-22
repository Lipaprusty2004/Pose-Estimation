import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def run_pose_estimation(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video source: {video_source}")
        return

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            cv2.imshow('Pose Estimation', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def upload_video():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if file_path:
        threading.Thread(target=run_pose_estimation, args=(file_path,), daemon=True).start()

def use_webcam():
    threading.Thread(target=run_pose_estimation, args=(0,), daemon=True).start()

root = tk.Tk()
root.title("Pose Estimation")
root.geometry("400x200")

main_heading = tk.Label(root, text="POSE ESTIMATION", font=("Helvetica", 18, "bold"))
main_heading.pack(pady=10)

upload_btn = tk.Button(root, text="Upload Video", command=upload_video, width=20)
upload_btn.pack(pady=10)

webcam_btn = tk.Button(root, text="Use Webcam", command=use_webcam, width=20)
webcam_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.destroy, width=20)
exit_btn.pack(pady=10)

root.mainloop()