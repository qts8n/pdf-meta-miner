import json
import sys

import cv2
import fitz
import numpy as np
from tika import parser

_PDF_DPI = 144
_THR_TOL = 3  # in pixels
_DTYPE = np.dtype('uint8')
_DTYPE.newbyteorder('=')

_TIKA_ENDPOINT = 'http://localhost:9998/tika/text/'
_TIKA_HEADERS = {'X-Tika-OCRLanguage': 'rus'}


def _print_usage():
    sys.stderr.write(f'USAGE: {__file__} <pdf_path>')


def _find_top_box(cnt_boxes, reference_box):
    top_left_box = None
    for cnt_box in cnt_boxes:
        tx, _, _, by = cnt_box
        if abs(tx - reference_box[0]) < _THR_TOL \
                and abs(by - reference_box[1]) < _THR_TOL:
            top_left_box = cnt_box
            break
    return top_left_box


def _find_left_box(cnt_boxes, reference_box):
    left_box = None
    for cnt_box in cnt_boxes:
        _, ty, bx, by = cnt_box
        if abs(bx - reference_box[0]) < _THR_TOL \
                and abs(ty - reference_box[1]) < _THR_TOL \
                and abs(by - reference_box[3]) < _THR_TOL:
            left_box = cnt_box
            break
    return left_box


def _find_corner_box(cnt_boxes, reference_box):
    corner_box = None
    for cnt_box in cnt_boxes:
        _, _, bx, by = cnt_box
        if cnt_box != reference_box \
                and abs(bx - reference_box[2]) < _THR_TOL \
                and abs(by - reference_box[3]) < _THR_TOL:
            corner_box = cnt_box
            break
    return corner_box


def find_metadata_around_cnt(cnt_boxes, biggest_cnt_idx):
    biggest_box = cnt_boxes[biggest_cnt_idx]

    # Find corner (signature) box
    corner_box = _find_corner_box(cnt_boxes, biggest_box)
    if corner_box is None:
        return None

    # Find scheme box
    scheme_box = _find_left_box(cnt_boxes, corner_box)
    if scheme_box is None:
        return None

    # Find section box
    section_box = _find_top_box(cnt_boxes, scheme_box)
    if section_box is None:
        return None

    # Find building box
    building_box = _find_top_box(cnt_boxes, section_box)
    if building_box is None:
        return None

    # Find project box
    project_box = _find_top_box(cnt_boxes, building_box)
    if project_box is None:
        return None

    # Find stage box
    stage_box = _find_top_box(cnt_boxes, corner_box)
    if stage_box is None:
        return None

    return {
        'project': project_box,
        'building': building_box,
        'section': section_box,
        'stage': stage_box,
        'scheme': scheme_box,
        'signature': corner_box
    }


def parse_metadata(bin_image):
    # Extract contour list
    contours, _ = cv2.findContours(
        bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there is enough contours to extract meta
    if len(contours) < 10:
        return None

    # Compute bounding boxes and find 2 biggest contours
    cnt_boxes = []
    biggest_cnt_idx, second_biggest_cnt_idx = -1, -1
    biggest_cnt, second_biggest_cnt = 0, 0
    for cnt_idx, cnt in enumerate(contours):
        cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(cnt)
        cnt_boxes.append((cnt_x, cnt_y, cnt_x + cnt_w, cnt_y + cnt_h))
        measure = cv2.contourArea(cnt)
        if measure > biggest_cnt:
            second_biggest_cnt = biggest_cnt
            second_biggest_cnt_idx = biggest_cnt_idx
            biggest_cnt = measure
            biggest_cnt_idx = cnt_idx
        elif measure > second_biggest_cnt:
            second_biggest_cnt = measure
            second_biggest_cnt_idx = cnt_idx
    meta_contours = find_metadata_around_cnt(cnt_boxes, biggest_cnt_idx)
    if meta_contours is None:
        meta_contours = find_metadata_around_cnt(cnt_boxes, second_biggest_cnt_idx)
    return meta_contours


def page_to_image(page):
    page_pixmap = page.get_pixmap(dpi=_PDF_DPI)
    c_channel_num = 4 if page_pixmap.alpha else 3
    shape = page_pixmap.height, page_pixmap.width, c_channel_num
    np_buffer = np.frombuffer(page_pixmap.samples, dtype=_DTYPE)
    image = np.reshape(np_buffer, shape)
    if c_channel_num == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def page_image_to_meta(page_image, line_min_width=50):
    gray_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
    thr_image = np.zeros_like(gray_image)
    thr_image[gray_image < 128] = 255
    kernel_h = np.ones((1, line_min_width), dtype=np.uint8)
    kernel_v = np.ones((line_min_width, 1), dtype=np.uint8)
    final_kernel = np.ones((3, 3), dtype=np.uint8)

    thr_image_h = cv2.morphologyEx(thr_image, cv2.MORPH_OPEN, kernel_h)
    thr_image_v = cv2.morphologyEx(thr_image, cv2.MORPH_OPEN, kernel_v)
    thr_image_final = cv2.dilate(thr_image_h | thr_image_v, final_kernel, iterations=1)
    metadata_boxes = parse_metadata(thr_image_final)
    metadata = {}
    for box_type, box in metadata_boxes.items():
        crop = page_image[box[1]:box[3], box[0]:box[2], :]
        cv2.imwrite(f'./assets/{box_type}.jpg', crop)
        crop_bytes = cv2.imencode('.jpg', crop)[1].tobytes()
        response = parser.from_buffer(crop_bytes, _TIKA_ENDPOINT, headers=_TIKA_HEADERS)
        content = response['content']
        if content is not None:
            content = content.strip().replace('\n', ' ')
        metadata[box_type] = content
    return metadata


_PDF_PAGE = 3


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        return 1
    pdf_path = argv[0]
    document = fitz.open(pdf_path)

    page = document.load_page(_PDF_PAGE)
    page_image = page_to_image(page)
    cv2.imwrite(f'./assets/{_PDF_PAGE}.jpg', page_image)
    metadata = page_image_to_meta(page_image)
    print(metadata)

    return 0


if __name__ == '__main__':
    status_code = main(sys.argv[1:])
    if status_code == 1:
        _print_usage()
    sys.exit(status_code)
