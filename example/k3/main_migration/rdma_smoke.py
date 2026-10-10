"""Tie each successful HTTP protocol request to the actual PD cache handoff."""

from request_smoke import require


def audit_protocol_handoff(result, prefill, decode):
    owners = max(1, len(result.get("decode_role_addrs", [])))
    require(len(decode) % owners == 0, "invalid Decode TP/DP layout")
    owner_width = len(decode) // owners
    identifiers = {}
    for events in prefill.values():
        for event in events:
            if event.get("event") == "frontend_request":
                identifiers[event["case"]] = event["request_id"]
    loaded = {
        rank: {
            event["request_id"]
            for event in events
            if event.get("event") == "pd_cache_loaded"
        }
        for rank, events in decode.items()
    }
    checked = []
    for record in result.get("protocol_requests", []):
        if record["transport"] != "openai" or record.get("expected_error"):
            continue
        name = record["name"]
        require(
            name in identifiers, f"protocol request has no frontend identity: {name}"
        )
        request = identifiers[name]
        owner_ranks = [rank for rank in decode if rank < owner_width]
        require(
            owner_ranks and all(request in loaded[rank] for rank in owner_ranks),
            f"protocol request lacks all-rank Decode cache-load completion: {name}",
        )
        checked.append({"case": name, "request_id": request})
    require(checked, "no protocol PD handoff was checked")
    return checked
