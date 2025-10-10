from mlcoe_q1.pipelines import extract_pdf_ratios as extractor


def test_parse_value_handles_currency_and_parentheses():
    assert extractor.parse_value("$1,200", 1.0) == 1200.0
    assert extractor.parse_value("(3,500)", 1.0) == -3500.0
    assert extractor.parse_value(None, 1.0) is None


def test_extract_number_prefers_explicit_column():
    row = ["Revenue", "1,000", "2,000"]
    assert extractor.extract_number(row, 1.0, preferred_index=1) == 1000.0


def test_extract_number_falls_back_to_regex():
    row = ["Total Assets note 5", "—", ""]
    value = extractor.extract_number(row, 1.0)
    assert value == 5.0


def test_label_matches_regex_and_literals():
    assert extractor.label_matches("Total Assets", ["total assets"]) is True
    assert extractor.label_matches("Total Assets", ["Total.*Assets"]) is True
    assert extractor.label_matches("Total Assets", ["Revenue"]) is False
