"""Smoke test for entity normalizer."""
from ingestion.entity_normalizer import normalize_entity, normalize_entities

print("=== Entity Normalizer Test ===")

# Should normalize
tests = [
    ("US", "United States"),
    ("UK", "United Kingdom"),
    ("UAE", "United Arab Emirates"),
    ("ukraine", "Ukraine"),
    ("DPRK", "North Korea"),
    ("  Cairo  ", "Cairo"),
]
for raw, expected in tests:
    result = normalize_entity(raw)
    status = "OK" if result == expected else f"FAIL (got {result})"
    print(f"  {raw!r:20s} -> {result!r:25s} {status}")

print()

# Should reject
rejects = ["AP", "NATO", "BBC", "2026", "A", "", "###", "http://example.com", "BREAKING"]
for raw in rejects:
    result = normalize_entity(raw)
    status = "OK (rejected)" if result is None else f"FAIL (got {result})"
    print(f"  {raw!r:20s} -> {result!r:10s} {status}")

print()

# Batch test
batch = ["Ukraine", "AP", "NATO", "Iran", "US", "2026", "BBC", "Gaza", "yemen"]
result = normalize_entities(batch)
print(f"Batch: {batch}")
print(f"  -> {result}")
print(f"  Filtered {len(batch) - len(result)} garbage tokens")
print()
print("Entity Normalizer: OK")
