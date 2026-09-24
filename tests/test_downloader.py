"""Unit tests for asset downloader helpers."""

from parallax.data.downloader import is_valid_image_bytes


def test_is_valid_image_bytes():
    # JPEG signature
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 30
    assert is_valid_image_bytes(jpeg_bytes) is True

    # PNG signature
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30
    assert is_valid_image_bytes(png_bytes) is True

    # Short / HTML bytes
    html_bytes = b"<html><head><title>404</title></head></html>"
    assert is_valid_image_bytes(html_bytes) is False

    empty_bytes = b""
    assert is_valid_image_bytes(empty_bytes) is False
