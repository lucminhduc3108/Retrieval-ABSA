from lxml import etree


def parse_semeval_xml(path: str) -> list[dict]:
    tree = etree.parse(path)
    root = tree.getroot()
    results = []
    for sentence in root.iter("sentence"):
        sid = sentence.get("id")
        text = sentence.findtext("text")
        opinions = []
        opinions_el = sentence.find("Opinions")
        if opinions_el is not None:
            for op in opinions_el.findall("Opinion"):
                if op.get("polarity") == "conflict":
                    continue
                target = op.get("target")
                opinions.append({
                    "target": None if target == "NULL" else target,
                    "category": op.get("category"),
                    "polarity": op.get("polarity"),
                    "from_char": int(op.get("from")),
                    "to_char": int(op.get("to")),
                })
        results.append({
            "sentence_id": sid,
            "text": text,
            "opinions": opinions,
        })
    return results


def parse_semeval2014_xml(path: str) -> list[dict]:
    tree = etree.parse(path)
    root = tree.getroot()
    results = []
    for sentence in root.iter("sentence"):
        sid = sentence.get("id")
        text = sentence.findtext("text")
        opinions = []
        cats_el = sentence.find("aspectCategories")
        if cats_el is not None:
            for ac in cats_el.findall("aspectCategory"):
                if ac.get("polarity") == "conflict":
                    continue
                opinions.append({
                    "target": None,
                    "category": ac.get("category"),
                    "polarity": ac.get("polarity"),
                    "from_char": 0,
                    "to_char": 0,
                })
        results.append({
            "sentence_id": sid,
            "text": text,
            "opinions": opinions,
        })
    return results
