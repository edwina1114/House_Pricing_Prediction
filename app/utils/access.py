from pathlib import Path
from PIL import Image


def get_data(filename: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / filename
    return data_path


def get_image(folder: str, filename: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    img_path = project_root / "images" / folder / filename
    img = Image.open(img_path)
    return img
