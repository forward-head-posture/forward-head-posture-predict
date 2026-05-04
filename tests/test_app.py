import numpy as np
import pytest
from predict.app import preprocess_frame, main


def test_preprocess_frame_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.shape == (1, 224, 224, 3)


@pytest.mark.parametrize("pixel,expected", [(0, -1.0), (255, 1.0)])
def test_preprocess_frame_normalization(pixel, expected):
    frame = np.full((480, 640, 3), pixel, dtype=np.uint8)
    result = preprocess_frame(frame)
    assert np.allclose(result, expected)


def test_main_reads_one_frame_and_exits(mocker):
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([[0.9]])

    mock_cap = mocker.MagicMock()
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mocker.patch("predict.app.cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("predict.app.cv2.imshow")
    mocker.patch("predict.app.cv2.waitKey", return_value=ord("q"))

    main(mock_model)

    mock_model.predict.assert_called_once()
