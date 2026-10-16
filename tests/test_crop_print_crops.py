from iwr.crops import CROP_LIBRARY
from iwr.crops import Crop


def test_crop_print_crops_prints_all_predefined_names(capsys) -> None:
    Crop.print_crops()

    captured = capsys.readouterr()
    expected = "\n".join(sorted(CROP_LIBRARY)) + "\n"

    assert captured.out == expected


def test_crop_print_crop_parameters_prints_maize_parameters(capsys) -> None:
    Crop.print_crop_parameters("maize")

    captured = capsys.readouterr()
    assert captured.out == (
        "name: maize\n"
        "root_max_depth: 1.7\n"
        "kc_initial: 0.3\n"
        "kc_max: 1.2\n"
        "kc_sen: 0.6\n"
        "p: 0.55\n"
    )