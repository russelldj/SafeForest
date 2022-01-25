import cv2
from mmseg.apis import inference_segmentor, init_segmentor
from typing import Union

from config import RUI_YAMAHA_PALETTE


def produce_segmentations(
    config: str,
    checkpoint: str,
    video_file: str,
    device: str = "cuda:0",
    opacity: float = 0.5,
    return_vis: bool = True,
):
    """A generator which yields predictions from a segmentation model

    Args:
        config: str,
        checkpoint: str,
        video_file: str,
        device: str,
        opacity: float = 0.5,
        return_vis: bool = True,
        return_probabilities: bool = False,

    Returns:
        A dict containing the IDs of the predicted image and optionaly the visualization
    """
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, checkpoint, device=device)

    # build input video
    cap = cv2.VideoCapture(str(video_file))

    # start looping
    while True:
        flag, frame = cap.read()
        if not flag:
            break

        result = inference_segmentor(model, frame)
        print(f"Result {result}")
        if return_vis:
            vis_img = model.show_result(
                frame, result, palette=RUI_YAMAHA_PALETTE, show=False, opacity=opacity
            )
            yield {"IDs": result[0], "vis": vis_img}
        else:
            yield {"IDs": result[0]}

    cap.release()
