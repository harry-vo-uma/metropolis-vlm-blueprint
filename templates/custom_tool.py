"""Add your own tool.

The registry validates the JSON Schema against the function signature at import
time, so a typo here is an ImportError rather than a confusing model failure at
runtime. Try renaming `zone` in one place but not the other to see it.

Run: python templates/custom_tool.py
"""

from mvb.tools import registry


@registry.register(
    name="zone_occupancy",
    description="Report how busy a named zone is over a time window.",
    parameters={
        "type": "object",
        "properties": {
            "zone": {"type": "string", "description": "Zone name, e.g. 'the loading dock'."},
            "window_s": {"type": "integer", "description": "Look-back window in seconds."},
        },
        "required": ["zone"],
    },
    tags=["analytics"],
)
def zone_occupancy(zone: str, window_s: int = 300) -> dict:
    from mvb.tools.builtin import search_events

    hits = search_events(query=zone, limit=50)
    return {"zone": zone, "window_s": window_s, "observations": len(hits)}


if __name__ == "__main__":
    print(registry.names())
    print(registry.call("zone_occupancy", {"zone": "the loading dock"}).content)
