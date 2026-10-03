"""Output JSON structure (our own schema; the organizers' published schema was
not in the provided material, see README)."""
SCHEMA_VERSION = "0.1"


def empty_result(capture_name, tier, title=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "capture": capture_name,
        "tier": tier,
        "title": title or capture_name,
        "rooms": [],
        "adjacency": [],
        "meta": {},
    }
