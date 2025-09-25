import cv2
import mediapipe as mp
import tempfile
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Mediapipe setup
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# ---------------- Processing Functions ----------------
def process_video(video_path, placeholder):
    count = 0
    position = None
    cap = cv2.VideoCapture(video_path)

    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Extract landmarks
                h, w, _ = image.shape
                imlist = []
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    imlist.append([id, int(lm.x * w), int(lm.y * h)])

                # sit-up logic
                if len(imlist) >= 15:
                    if imlist[26][2] >= imlist[24][2] and imlist[25][2] >= imlist[23][2]:
                        position = "down"
                    if position == "down" and imlist[26][2] < imlist[24][2] and imlist[25][2] < imlist[23][2]:
                        position = "up"
                        count += 1

            cv2.putText(image, f'sit-ups: {count}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
    return count


# ---------------- Live Video Processor ----------------
class PushupCounter(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.position = None
        self.pose = mp_pose.Pose(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.7)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            h, w, _ = image.shape
            imlist = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                imlist.append([id, int(lm.x * w), int(lm.y * h)])

            if len(imlist) >= 15:
                if imlist[26][2] >= imlist[24][2] and imlist[25][2] >= imlist[23][2]:
                    self.position = "down"
                if self.position == "down" and imlist[26][2] < imlist[24][2] and imlist[25][2] < imlist[23][2]:
                    self.position = "up"
                    self.count += 1

        cv2.putText(image, f'sit-ups: {self.count}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ---------------- Streamlit UI ----------------
def main():
    st.title("🏋️ sit-up Counter with Mediapipe")

    mode = st.radio("Choose mode:", ["📹 Upload Video", "🎥 Live Camera"])

    if mode == "📹 Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.info("Processing video... ⏳")
            placeholder = st.empty()
            total_count = process_video(tfile.name, placeholder)
            st.success(f"✅ Done! Total sit-ups counted: {total_count}")

    elif mode == "🎥 Live Camera":
        st.info("Live mode started... do sit-ups in front of your webcam 🎥")
        webrtc_streamer(
            key="pushup-counter",
            video_processor_factory=PushupCounter,
            media_stream_constraints={"video": True, "audio": False},
        )


if __name__ == "__main__":
    main()
