"""Handshake conveys a portable usage sequence without loading target knowledge."""
import pytest

from skillforge.integration.mcp_server import SERVER_INSTRUCTIONS, dispatch


@pytest.mark.parametrize("version", ["2024-11-05", "2025-03-26", "2025-06-18", "unknown"])
def test_initialize_delivers_usage_without_touching_knowledge(version):
    class MetadataOnly:
        def call(self, *_):
            pytest.fail("initialize must not load or route knowledge")

    response = dispatch(MetadataOnly(), {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": version, "capabilities": {},
                   "clientInfo": {"name": "ordinary-client", "version": "1"}},
    })
    result = response["result"]
    assert result["capabilities"] == {"tools": {}}
    assert result["instructions"] == SERVER_INSTRUCTIONS
    assert result["protocolVersion"] == ("2025-06-18" if version == "unknown" else version)
    first = result["instructions"][:512]
    assert first.index("get_related_workflow") < first.index("get_skill") < first.index("verify_patch")
    assert "incomplete" in first
    assert "skill-" not in result["instructions"]
