"""Per-request log identifiers and best-effort client source classification."""
from __future__ import annotations

import ipaddress
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Mapping

_source: ContextVar[str] = ContextVar("request_source", default="-")
_client_id: ContextVar[str] = ContextVar("client_id", default="-")
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def current_request_context() -> dict[str, str]:
    return {
        "source": _source.get(),
        "client_id": _client_id.get(),
        "request_id": _request_id.get(),
    }


@contextmanager
def bind_request_context(
    source: str,
    client_id: str,
    request_id: str,
) -> Iterator[None]:
    tokens: tuple[Token[str], Token[str], Token[str]] = (
        _source.set(source),
        _client_id.set(client_id),
        _request_id.set(request_id),
    )
    try:
        yield
    finally:
        _source.reset(tokens[0])
        _client_id.reset(tokens[1])
        _request_id.reset(tokens[2])


def classify_client_source(
    client_ip: str | None,
    headers: Mapping[str, str] | None = None,
) -> str:
    """Classify a local/private client versus public client without retaining its IP.

    Cloudflare's visitor IP is used only when Cloudflare also supplied a Ray ID.
    This is an operational label, not a security or authentication decision.
    """
    headers = headers or {}
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    forwarded_ip = normalized_headers.get("cf-connecting-ip")
    if forwarded_ip and normalized_headers.get("cf-ray"):
        client_ip = forwarded_ip

    if not client_ip:
        return "unknown"
    try:
        address = ipaddress.ip_address(client_ip.strip())
    except ValueError:
        return "unknown"
    return "internal" if address.is_private or address.is_loopback else "external"
