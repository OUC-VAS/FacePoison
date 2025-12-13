import cv2
import os

# False ---  inference
videofacepoison = True

if videofacepoison:
    data_path = '/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/cdfv2_video'
    output_frame = "/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/sample_img_data_5frame"
else:   # for testing_deepfake_path
    data_path = '/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/inference_deepfake_videos'
    output_frame = "/media/hash/data/code/face_poison/VideoFacePosion/attack_public_datasets/inference_sample_img_data_5frame"


def extract_frames11(output_path, video_path, video_name):        # video2frame-5frame
    os.makedirs(output_path, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frame_count = 0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs = []

    while True:
        success, frame = video_capture.read()
        if success:
            imgs.append(frame)

            if frame_count % 5 == 0:
                frame_name = video_name + "_" + f"{frame_count:04d}.png"
                frame_path = os.path.join(output_path, frame_name)

                cv2.imwrite(frame_path, frame)
                saved_frame_count += 1
        else:
            break
        frame_count += 1

    video_capture.release()
def extract_frames(dataset_name, output_path, video_path, video_name):        # video2frame-5frame
    os.makedirs(output_path, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frame_count = 0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs = []

    while True:
        success, frame = video_capture.read()
        if success:
            imgs.append(frame)

            if frame_count % 5 == 0:
                frame_name = video_name + "_" + f"{frame_count:04d}.png"
                frame_path = os.path.join(output_path, frame_name)

                # resize img: (1920*1080 or 1080*1920) to (960*540 or 540*960)
                if dataset_name == 'DFDCP' or dataset_name == 'DFDC':
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(frame_path, frame)
                saved_frame_count += 1
        else:
            break
        frame_count += 1

    video_capture.release()

if videofacepoison:
    for video in sorted(os.listdir(data_path)):     # cfdv2
        extract_frames11(output_path=os.path.join(output_frame, video.split('_')[0]), video_path=os.path.join(data_path, video), video_name=video[:-4])
else:
    for dataset_name in os.listdir(data_path):
        for video in sorted(os.listdir(os.path.join(data_path, dataset_name))):

            if dataset_name == 'FF++' or dataset_name == 'DFDC':
                output_path = os.path.join(output_frame, dataset_name, video[:-4])
            else:
                output_path = os.path.join(output_frame, dataset_name, video.split('_')[0])
            extract_frames(dataset_name, output_path, video_path=os.path.join(data_path, dataset_name, video), video_name=video[:-4])
