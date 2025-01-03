import cv2
import imageio

def make_video(video_infos):
    # 비디오 캡처 객체 생성
    caps = [cv2.VideoCapture(path) for path in video_infos['file_paths']]

    # 모든 비디오의 해상도와 FPS를 동일하게 설정
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    # 모든 비디오 해상도를 기준 해상도로 변환
    def resize_frame(frame, width, height):
        return cv2.resize(frame, (width, height))

    # 타이틀 추가 함수
    def add_title_to_frame(frame, title):
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = 40  # 타이틀 위치
        # 배경 박스
        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (50, 50, 50),  # 짙은 회색
            -1  # 채우기
        )
        # 타이틀 텍스트
        cv2.putText(
            overlay,
            title,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # 흰색
            font_thickness,
            cv2.LINE_AA
        )
        return overlay

    # 출력 비디오 설정
    output_width = frame_width * len(caps)
    output_height = frame_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_infos['output_file_path'], fourcc, fps, (output_width, output_height))

    while True:
        frames = []
        for cap, title in zip(caps, video_infos['file_title']):
            ret, frame = cap.read()
            if not ret:
                print("End of one or more videos.")
                break
            frame = resize_frame(frame, frame_width, frame_height)
            frame_with_title = add_title_to_frame(frame, title)
            frames.append(frame_with_title)

        if len(frames) != len(caps):
            break

        # 프레임을 가로로 합치기
        combined_frame = cv2.hconcat(frames)
        out.write(combined_frame)

    # 자원 해제
    for cap in caps:
        cap.release()
    out.release()

    print(f"Combined video with titles saved at: {video_infos['output_file_path']}")

def convert_mp4_to_moving_gif(mp4_file, gif_file, save_image_scale=0.5):
    cap = cv2.VideoCapture(mp4_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        new_size = (int(width * save_image_scale), int(height * save_image_scale))
        frame = cv2.resize(frame, new_size)
        frames.append(frame)
    cap.release()
    imageio.mimsave(gif_file, frames, fps=15, loop=0)
    print(f"GIF saved at: {gif_file}")


if __name__ == "__main__":
    video_infos = {
        'file_paths': [  # 비디오 파일 경로
            "../exp/pvg_example/videos_eval/full_set_30000_gt_rgbs.mp4",
            "../exp/pvg_example/videos_eval/full_set_30000_rgbs.mp4",
            "../exp/pvg_example/videos_eval/novel_30000/change_lane_left.mp4"
        ],
        'file_title': [  # 비디오 타이틀
            "GT",
            "Novel View (DriveDreamer4D with PVG)",
            "Change Left Lane (DriveDreamer4D with PVG)"
        ],
        'output_file_path':  # 출력 파일 경로
            "../exp/pvg_example/videos_eval/pvg_example.mp4"
    }
    make_video(video_infos=video_infos)

    convert_mp4_to_moving_gif(mp4_file=video_infos['output_file_path'],
                              gif_file='../exp/pvg_example/videos_eval/pvg_example.gif')
