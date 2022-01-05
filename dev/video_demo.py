from argparse import ArgumentParser

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

from config import RUI_CLASSES


def main():
    parser = ArgumentParser()
    parser.add_argument("video", help="Video file or webcam id")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--palette",
        default="cityscapes",
        help="Color palette used for segmentation map",
    )
    parser.add_argument(
        "--show", action="store_true", help="Whether to show draw result"
    )
    parser.add_argument(
        "--show-wait-time", default=1, type=int, help="Wait time after imshow"
    )
    parser.add_argument(
        "--output-file", default=None, type=str, help="Output video file path"
    )
    parser.add_argument(
        "--output-fourcc", default="MJPG", type=str, help="Fourcc of the output video"
    )
    parser.add_argument(
        "--output-fps", default=-1, type=int, help="FPS of the output video"
    )
    parser.add_argument(
        "--output-height", default=-1, type=int, help="Frame height of the output video"
    )
    parser.add_argument(
        "--output-width", default=-1, type=int, help="Frame width of the output video"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    args = parser.parse_args()

    assert args.show or args.output_file, "At least one output should be enabled."

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # build input video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # init output video
    writer = None
    output_height = None
    output_width = None
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = (
            args.output_height if args.output_height > 0 else int(input_height)
        )
        output_width = args.output_width if args.output_width > 0 else int(input_width)
        writer = cv2.VideoWriter(
            args.output_file, fourcc, output_fps, (output_width, output_height), True
        )

    # start looping
    try:
        i = 0
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            # test a single image
            if i % 100 == 0:
                result = inference_segmentor(model, frame, return_probabilities=True)[0]
                fig, axes = plt.subplots(2, 4)
                for i in range(2):
                    for j in range(3):
                        axes[i, j].imshow(result[..., 3 * i + j])
                        axes[i, j].set_title(RUI_CLASSES[3 * i + j])
                axes[0, 3].imshow(np.flip(frame, axis=2))
                axes[0, 3].set_title("Image")
                plt.show()
            i += 1

    finally:
        if writer:
            writer.release()
        cap.release()


if __name__ == "__main__":
    main()
