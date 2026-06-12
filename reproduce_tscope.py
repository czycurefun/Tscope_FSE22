from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_client import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_ENV,
    DEFAULT_API_KEY_FILE,
    DEFAULT_MODEL,
    RemoteChatClient,
    resolve_api_key,
)


ENTITY_TYPES = ["Component", "Behavior", "Prerequisite", "Manner", "Constraint"]
RELATION_TYPES = {
    ("Component", "Behavior"): "Act",
    ("Component", "Prerequisite"): "Require",
    ("Component", "Manner"): "Use",
    ("Component", "Constraint"): "Satisfy",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "case",
    "each",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "result",
    "shall",
    "should",
    "system",
    "test",
    "the",
    "then",
    "to",
    "using",
    "verify",
    "when",
    "with",
}

BEHAVIOR_SYNONYMS = {
    "view": "browse",
    "display": "browse",
    "show": "browse",
    "open": "open",
    "opening": "open",
    "switch": "switch",
    "browse": "browse",
    "test": "test",
    "verify": "verify",
    "check": "verify",
    "validate": "verify",
}

BEHAVIOR_WORDS = set(BEHAVIOR_SYNONYMS) | {
    "add",
    "cancel",
    "click",
    "delete",
    "download",
    "edit",
    "export",
    "filter",
    "install",
    "login",
    "print",
    "register",
    "remove",
    "run",
    "search",
    "select",
    "sort",
    "upload",
}

LOGIC_INDICATORS = {"no", "not", "without", "except", "only"}
TEMPORAL_INDICATORS = {"after", "before", "when", "while", "during", "until"}


@dataclass
class Entity:
    id: int
    type: str
    value: str
    start: int
    end: int


@dataclass
class Relation:
    type: str
    head: int
    tail: int
    head_value: str
    tail_value: str


@dataclass
class TestTuple:
    Component: str = "NULL"
    Behavior: str = "NULL"
    Prerequisite: str = "NULL"
    Manner: str = "NULL"
    Constraint: str = "NULL"


@dataclass
class Extraction:
    id: str
    text: str
    tokens: list[str]
    entities: list[Entity]
    relations: list[Relation]
    tuples: list[TestTuple] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.;:")


def normalize_phrase(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = normalize_space(text)
    return text


def strip_stopwords(text: str) -> list[str]:
    tokens = tokenize(text)
    return [BEHAVIOR_SYNONYMS.get(token, token) for token in tokens if token not in STOPWORDS]


def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def canonical_tokens(text: str) -> list[str]:
    return [singularize(token) for token in strip_stopwords(text)]


def find_span(text: str, phrase: str) -> tuple[int, int]:
    pattern = re.escape(phrase)
    match = re.search(pattern, text, flags=re.I)
    if match:
        return match.start(), match.end()
    return -1, -1


def clean_component_phrase(phrase: str) -> str:
    phrase = normalize_phrase(phrase)
    phrase = re.sub(r"^(to|the|a|an)\s+", "", phrase)
    phrase = re.split(
        r"\b(after|before|when|while|using|with|via|by|then|and|including|without|must|should|shall)\b",
        phrase,
    )[0]
    phrase = normalize_space(phrase)
    return phrase


def unique_entities(entities: list[Entity]) -> list[Entity]:
    seen: set[tuple[str, str]] = set()
    result: list[Entity] = []
    next_id = 0
    for entity in entities:
        value = normalize_phrase(entity.value)
        if not value or (entity.type, value) in seen:
            continue
        seen.add((entity.type, value))
        result.append(Entity(next_id, entity.type, value, entity.start, entity.end))
        next_id += 1
    return result


def extract_prerequisites(text: str) -> list[str]:
    clauses: list[str] = []
    for match in re.finditer(r"\b(when|while|after|before|during|if)\b([^,.]+)", text, flags=re.I):
        clause = normalize_space(match.group(0))
        if len(tokenize(clause)) >= 2:
            clauses.append(clause)
    return clauses


def extract_manners(text: str) -> list[str]:
    manners: list[str] = []
    for match in re.finditer(r"\b(using|with|via|by|through)\b\s+([^,.]+)", text, flags=re.I):
        phrase = match.group(0)
        phrase = re.split(r"\b(after|before|when|while|then|to|for)\b", phrase, flags=re.I)[0]
        phrase = re.sub(r"^(using|with|via|by|through)\s+", "", phrase, flags=re.I)
        phrase = normalize_space(phrase)
        if phrase:
            manners.append(phrase)
    return manners


def extract_constraints(text: str) -> list[str]:
    constraints: list[str] = []
    patterns = [
        r"\bincluding\b\s+([^,.]+)",
        r"\bwithout\b\s+([^,.]+)",
        r"\bat least\b\s+([^,.]+)",
        r"\bno more than\b\s+([^,.]+)",
        r"\bonly\b\s+([^,.]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            constraints.append(normalize_space(match.group(0)))
    return constraints


def extract_behaviors(text: str) -> list[str]:
    tokens = tokenize(text)
    behaviors: list[str] = []
    for token in tokens:
        if token in BEHAVIOR_WORDS:
            behaviors.append(BEHAVIOR_SYNONYMS.get(token, token))
    return behaviors


def extract_components(text: str, behaviors: list[str]) -> list[str]:
    components: list[str] = []
    behavior_pattern = "|".join(sorted(re.escape(word) for word in set(BEHAVIOR_WORDS)))
    for match in re.finditer(rf"\b({behavior_pattern})\b\s+(?:to\s+)?([^,.]+)", text, flags=re.I):
        phrase = clean_component_phrase(match.group(2))
        if phrase and phrase not in BEHAVIOR_SYNONYMS:
            components.append(phrase)

    # Additional component candidates from explicit noun-like anchors often used in test cases.
    anchors = [
        r"visit history",
        r"contents? of each resource directory",
        r"resource directory",
        r"gear rotation processing",
        r"preset applications?",
        r"ftp application",
    ]
    for pattern in anchors:
        for match in re.finditer(pattern, text, flags=re.I):
            components.append(normalize_phrase(match.group(0)))

    cleaned: list[str] = []
    for component in components:
        component = re.sub(r"^(the|a|an)\s+", "", component)
        component = normalize_space(component)
        if component and component not in cleaned:
            cleaned.append(component)
    return cleaned


def build_entities(text: str) -> list[Entity]:
    raw_entities: list[Entity] = []
    behaviors = extract_behaviors(text)
    components = extract_components(text, behaviors)
    prerequisites = extract_prerequisites(text)
    manners = extract_manners(text)
    constraints = extract_constraints(text)
    grouped = [
        ("Component", components),
        ("Behavior", behaviors),
        ("Prerequisite", prerequisites),
        ("Manner", manners),
        ("Constraint", constraints),
    ]
    eid = 0
    for entity_type, values in grouped:
        for value in values:
            start, end = find_span(text, value)
            raw_entities.append(Entity(eid, entity_type, value, start, end))
            eid += 1
    return unique_entities(raw_entities)


def distance_to_component(entity: Entity, component: Entity) -> int:
    if entity.start < 0 or component.start < 0:
        return 10**6
    return abs(entity.start - component.start)


def relation_allowed(component: Entity, entity: Entity) -> str | None:
    return RELATION_TYPES.get(("Component", entity.type))


def build_relations(entities: list[Entity]) -> list[Relation]:
    components = [entity for entity in entities if entity.type == "Component"]
    others = [entity for entity in entities if entity.type != "Component"]
    relations: list[Relation] = []
    for entity in others:
        relation_type = None
        if components:
            component = min(components, key=lambda item: distance_to_component(entity, item))
            relation_type = relation_allowed(component, entity)
            if relation_type:
                relations.append(
                    Relation(
                        type=relation_type,
                        head=entity.id,
                        tail=component.id,
                        head_value=entity.value,
                        tail_value=component.value,
                    )
                )
    return relations


def dissect_tuples(entities: list[Entity], relations: list[Relation]) -> list[TestTuple]:
    by_id = {entity.id: entity for entity in entities}
    components = [entity for entity in entities if entity.type == "Component"]
    if not components:
        return []
    grouped: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for relation in relations:
        tail_entity = by_id.get(relation.tail)
        head_entity = by_id.get(relation.head)
        if not tail_entity or not head_entity or tail_entity.type != "Component":
            continue
        grouped[tail_entity.id][head_entity.type].append(head_entity.value)

    tuples: list[TestTuple] = []
    for component in components:
        data = grouped[component.id]
        tuples.append(
            TestTuple(
                Component=component.value,
                Behavior=", ".join(data.get("Behavior", [])) or "NULL",
                Prerequisite=", ".join(data.get("Prerequisite", [])) or "NULL",
                Manner=", ".join(data.get("Manner", [])) or "NULL",
                Constraint=", ".join(data.get("Constraint", [])) or "NULL",
            )
        )
    return tuples


def extract_test_case(case: dict[str, Any]) -> Extraction:
    text = case["text"]
    entities = build_entities(text)
    relations = build_relations(entities)
    tuples = dissect_tuples(entities, relations)
    return Extraction(
        id=case["id"],
        text=text,
        tokens=tokenize(text),
        entities=entities,
        relations=relations,
        tuples=tuples,
    )


def parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    return json.loads(text[start : end + 1])


def make_llm_prompt(text: str) -> str:
    return f"""
Extract test-oriented entities and relations for Tscope redundant test-case detection.

Entity categories:
- Component: tested functional component or object.
- Behavior: action performed on a component.
- Prerequisite: pre-condition before execution.
- Manner: tool, interface, operation way, or execution manner.
- Constraint: requirement or constraint that must be satisfied.

Relation categories:
- Act: Component-Behavior relation.
- Require: Component-Prerequisite relation.
- Use: Component-Manner relation.
- Satisfy: Component-Constraint relation.

Return JSON only with this schema:
{{
  "entities": [
    {{"type": "Component|Behavior|Prerequisite|Manner|Constraint", "value": "short exact phrase"}}
  ],
  "relations": [
    {{
      "type": "Act|Require|Use|Satisfy",
      "component": "component phrase from entities",
      "related_type": "Behavior|Prerequisite|Manner|Constraint",
      "related": "related entity phrase from entities"
    }}
  ]
}}

Rules:
- Use short phrases from the test case whenever possible.
- Do not invent entities.
- Keep duplicate entities only once.
- If a component has multiple behaviors, output multiple Act relations.
- Phrases introduced by "including", "without", "only", "at least", or "no more than" must be extracted as Constraint when present.
- If no entity or relation exists, use an empty list.

Example:
Text: "Using the mouse, browse the visit history after opening the resource center."
Output:
{{
  "entities": [
    {{"type": "Component", "value": "visit history"}},
    {{"type": "Behavior", "value": "browse"}},
    {{"type": "Prerequisite", "value": "after opening the resource center"}},
    {{"type": "Manner", "value": "mouse"}}
  ],
  "relations": [
    {{"type": "Act", "component": "visit history", "related_type": "Behavior", "related": "browse"}},
    {{"type": "Require", "component": "visit history", "related_type": "Prerequisite", "related": "after opening the resource center"}},
    {{"type": "Use", "component": "visit history", "related_type": "Manner", "related": "mouse"}}
  ]
}}

Example:
Text: "After the system installation, verify preset applications including FTP application."
Output:
{{
  "entities": [
    {{"type": "Component", "value": "preset applications"}},
    {{"type": "Behavior", "value": "verify"}},
    {{"type": "Prerequisite", "value": "after the system installation"}},
    {{"type": "Constraint", "value": "including FTP application"}}
  ],
  "relations": [
    {{"type": "Act", "component": "preset applications", "related_type": "Behavior", "related": "verify"}},
    {{"type": "Require", "component": "preset applications", "related_type": "Prerequisite", "related": "after the system installation"}},
    {{"type": "Satisfy", "component": "preset applications", "related_type": "Constraint", "related": "including FTP application"}}
  ]
}}

Text: {json.dumps(text, ensure_ascii=False)}
Output:
""".strip()


def normalize_entity_type(value: str) -> str | None:
    normalized = value.strip().lower()
    mapping = {item.lower(): item for item in ENTITY_TYPES}
    return mapping.get(normalized)


def normalize_relation_type(value: str) -> str | None:
    normalized = value.strip().lower()
    mapping = {item.lower(): item for item in RELATION_TYPES.values()}
    return mapping.get(normalized)


def find_entity_id(entities: list[Entity], entity_type: str, value: str) -> int | None:
    target = normalize_phrase(value)
    candidates = [entity for entity in entities if entity.type == entity_type]
    for entity in candidates:
        if normalize_phrase(entity.value) == target:
            return entity.id
    for entity in candidates:
        left = set(canonical_tokens(entity.value))
        right = set(canonical_tokens(target))
        if left and right and (left <= right or right <= left):
            return entity.id
    return None


def extraction_from_llm(case: dict[str, Any], client: RemoteChatClient) -> Extraction:
    text = case["text"]
    raw = client.complete(
        make_llm_prompt(text),
        system_prompt="You are an information extraction engine. Return compact valid JSON only.",
    )
    parsed = parse_json_object(raw)

    raw_entities: list[Entity] = []
    for index, item in enumerate(parsed.get("entities", [])):
        if not isinstance(item, dict):
            continue
        entity_type = normalize_entity_type(str(item.get("type", "")))
        value = normalize_phrase(str(item.get("value", "")))
        if not entity_type or not value:
            continue
        start, end = find_span(text, value)
        raw_entities.append(Entity(index, entity_type, value, start, end))
    entities = unique_entities(raw_entities)

    relations: list[Relation] = []
    for item in parsed.get("relations", []):
        if not isinstance(item, dict):
            continue
        relation_type = normalize_relation_type(str(item.get("type", "")))
        component_value = str(item.get("component", ""))
        related_type = normalize_entity_type(str(item.get("related_type", "")))
        related_value = str(item.get("related", ""))
        if not relation_type or not related_type or related_type == "Component":
            continue
        component_id = find_entity_id(entities, "Component", component_value)
        related_id = find_entity_id(entities, related_type, related_value)
        if component_id is None or related_id is None:
            continue
        relations.append(
            Relation(
                type=relation_type,
                head=related_id,
                tail=component_id,
                head_value=entities[related_id].value,
                tail_value=entities[component_id].value,
            )
        )

    tuples = dissect_tuples(entities, relations)
    return Extraction(
        id=case["id"],
        text=text,
        tokens=tokenize(text),
        entities=entities,
        relations=relations,
        tuples=tuples,
    )


def cosine_from_tokens(left_tokens: list[str], right_tokens: list[str]) -> float:
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    left = Counter(left_tokens)
    right = Counter(right_tokens)
    vocab = set(left) | set(right)
    numerator = sum(left[token] * right[token] for token in vocab)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def contains_equivalent(left: str, right: str) -> bool:
    left_tokens = set(canonical_tokens(left))
    right_tokens = set(canonical_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    return left_tokens <= right_tokens or right_tokens <= left_tokens


def indicative_words(text: str) -> set[str]:
    tokens = set(tokenize(text))
    return (tokens & LOGIC_INDICATORS) | (tokens & TEMPORAL_INDICATORS)


def entity_similarity(left: str, right: str, entity_type: str) -> float:
    if left == "NULL" and right == "NULL":
        return 1.0
    if left == "NULL" or right == "NULL":
        return 0.0
    if entity_type == "Behavior":
        left_tokens = [BEHAVIOR_SYNONYMS.get(token, token) for token in canonical_tokens(left)]
        right_tokens = [BEHAVIOR_SYNONYMS.get(token, token) for token in canonical_tokens(right)]
        return cosine_from_tokens(left_tokens, right_tokens)
    if entity_type == "Prerequisite":
        left_indicators = indicative_words(left)
        right_indicators = indicative_words(right)
        if left_indicators != right_indicators:
            return 0.0
    if contains_equivalent(left, right):
        return 1.0
    return cosine_from_tokens(canonical_tokens(left), canonical_tokens(right))


def tuple_equivalent(left: TestTuple, right: TestTuple, threshold: float) -> tuple[bool, dict[str, float]]:
    scores = {
        field: entity_similarity(getattr(left, field), getattr(right, field), field)
        for field in ENTITY_TYPES
    }
    return all(score >= threshold for score in scores.values()), scores


def covers(left_tuples: list[TestTuple], right_tuples: list[TestTuple], threshold: float) -> tuple[bool, list[dict[str, Any]]]:
    if not right_tuples:
        return False, []
    details: list[dict[str, Any]] = []
    for right_index, right_tuple in enumerate(right_tuples):
        best = {"left_index": None, "right_index": right_index, "equivalent": False, "scores": {}, "average": 0.0}
        for left_index, left_tuple in enumerate(left_tuples):
            equivalent, scores = tuple_equivalent(left_tuple, right_tuple, threshold)
            average = sum(scores.values()) / len(scores)
            if average > best["average"]:
                best = {
                    "left_index": left_index,
                    "right_index": right_index,
                    "equivalent": equivalent,
                    "scores": scores,
                    "average": average,
                }
        details.append(best)
    return all(item["equivalent"] for item in details), details


def detect_redundancy(extractions: list[Extraction], threshold: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, left in enumerate(extractions):
        for j, right in enumerate(extractions):
            if i == j:
                continue
            covered, details = covers(left.tuples, right.tuples, threshold)
            if covered:
                results.append(
                    {
                        "covering_test_case": left.id,
                        "redundant_test_case": right.id,
                        "relation": "totally_equivalent"
                        if len(left.tuples) == len(right.tuples)
                        else "covered_by",
                        "tuple_match_details": details,
                    }
                )
    return results


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    cases = json.loads(input_path.read_text(encoding="utf-8"))
    if args.extractor == "llm":
        api_key_value = resolve_api_key(args.api_key, args.api_key_env, args.api_key_file, args.api_key_var)
        if not api_key_value:
            raise SystemExit("Missing API key. Set ANTCHAT_API_KEY or pass --api-key.")
        client = RemoteChatClient(
            api_base_url=args.api_base_url,
            api_key_value=api_key_value,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            use_env_proxy=args.use_env_proxy,
            request_sleep=args.request_sleep,
        )
        extractions = [extraction_from_llm(case, client) for case in cases]
    else:
        extractions = [extract_test_case(case) for case in cases]
    redundancy = detect_redundancy(extractions, args.threshold)

    extraction_json = []
    tuple_json = []
    for extraction in extractions:
        extraction_json.append(
            {
                "id": extraction.id,
                "text": extraction.text,
                "tokens": extraction.tokens,
                "entities": [asdict(entity) for entity in extraction.entities],
                "relations": [asdict(relation) for relation in extraction.relations],
            }
        )
        tuple_json.append(
            {
                "id": extraction.id,
                "tuples": [asdict(item) for item in extraction.tuples],
            }
        )

    summary = {
        "method": "Tscope reproduction: preprocess, extract test-oriented entities and relations, dissect tuples, detect redundancy by tuple covering",
        "input": str(input_path),
        "extractor": args.extractor,
        "llm": {
            "model": args.model if args.extractor == "llm" else "",
            "api_base_url": args.api_base_url if args.extractor == "llm" else "",
            "key_source": args.api_key_env if args.api_key else f"env:{args.api_key_env} or file:{args.api_key_file}",
        },
        "threshold": args.threshold,
        "test_case_count": len(cases),
        "redundant_pair_count": len(redundancy),
        "redundant_pairs": redundancy,
    }
    write_json(extraction_json, output_dir / "extractions.json")
    write_json(tuple_json, output_dir / "tuples.json")
    write_json(summary, output_dir / "redundancy_results.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved outputs to: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact Tscope reproduction.")
    parser.add_argument("--input", default="data/reproduction/sample_testcases.json")
    parser.add_argument("--output-dir", default="outputs/reproduction_tscope")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--extractor", choices=["llm", "rule"], default="llm")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--api-key-var", default="API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--use-env-proxy", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
