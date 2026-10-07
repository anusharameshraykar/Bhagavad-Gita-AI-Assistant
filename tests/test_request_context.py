import logging

from src.logging_config import configure_logging
from src.request_context import (
    bind_request_context,
    classify_client_source,
    current_request_context,
)


def test_client_source_classifies_local_and_public_addresses():
    assert classify_client_source("127.0.0.1") == "internal"
    assert classify_client_source("192.168.1.42") == "internal"
    assert classify_client_source("8.8.8.8") == "external"
    assert classify_client_source(None) == "unknown"
    assert classify_client_source("not-an-ip") == "unknown"


def test_cloudflare_visitor_ip_is_used_when_ray_header_is_present():
    assert classify_client_source(
        "127.0.0.1",
        {
            "CF-Connecting-IP": "1.1.1.1",
            "CF-Ray": "abc123-BOM",
        },
    ) == "external"


def test_request_context_is_restored_after_scope():
    initial = current_request_context()
    with bind_request_context("external", "client-123", "request-456"):
        assert current_request_context() == {
            "source": "external",
            "client_id": "client-123",
            "request_id": "request-456",
        }
    assert current_request_context() == initial


def test_log_formatter_includes_identifiers_without_client_ip():
    configure_logging()
    handler = logging.getLogger("src").handlers[0]
    record = logging.LogRecord(
        "src.test",
        logging.INFO,
        __file__,
        1,
        "Question received.",
        (),
        None,
    )
    with bind_request_context("external", "client-123", "request-456"):
        handler.filter(record)
        output = handler.format(record)

    assert "source=external" in output
    assert "client=client-123" in output
    assert "request=request-456" in output
    assert "Question received." in output
    assert "1.1.1.1" not in output
