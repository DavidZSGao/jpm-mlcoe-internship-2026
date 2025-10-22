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


def test_company_config_includes_alibaba_defaults():
    cfg = extractor.COMPANY_CONFIG['alibaba']
    assert 'income' in cfg and 'balance' in cfg
    assert any('operations' in label.lower() for label in cfg['income']['operating_income'])
    assert any('shareholders' in label.lower() for label in cfg['balance']['equity'])


def test_parse_args_accepts_alibaba(tmp_path):
    pdf_path = tmp_path / 'alibaba.pdf'
    pdf_path.write_bytes(b'')
    args = extractor.parse_args([str(pdf_path), '--company', 'alibaba'])
    assert args.company == 'alibaba'


def test_company_config_includes_new_presets():
    assert 'jpm' in extractor.COMPANY_CONFIG
    assert 'exxon' in extractor.COMPANY_CONFIG
    assert 'microsoft' in extractor.COMPANY_CONFIG
    assert 'vw' in extractor.COMPANY_CONFIG
    assert 'alphabet' in extractor.COMPANY_CONFIG
    assert 'google' in extractor.COMPANY_CONFIG
    assert 'mercedes' in extractor.COMPANY_CONFIG
    assert 'sap' in extractor.COMPANY_CONFIG
    assert 'toyota' in extractor.COMPANY_CONFIG
    assert 'nestle' in extractor.COMPANY_CONFIG
    assert 'hsbc' in extractor.COMPANY_CONFIG
    assert 'santander' in extractor.COMPANY_CONFIG
    jpm_cfg = extractor.COMPANY_CONFIG['jpm']
    assert jpm_cfg['income']['expense_mode'] == 'label'
    exxon_cfg = extractor.COMPANY_CONFIG['exxon']
    assert any('revenues' in label.lower() for label in exxon_cfg['income']['revenue'])
    msft_cfg = extractor.COMPANY_CONFIG['microsoft']
    assert 'Total current assets' in msft_cfg['balance']['current_assets'][0]
    vw_cfg = extractor.COMPANY_CONFIG['vw']
    assert any('operating' in label.lower() for label in vw_cfg['income']['operating_income'])
    toyota_cfg = extractor.COMPANY_CONFIG['toyota']
    assert any('net income' in label.lower() for label in toyota_cfg['income']['net_income'])
    nestle_cfg = extractor.COMPANY_CONFIG['nestle']
    assert any('trading operating' in label.lower() for label in nestle_cfg['income']['operating_income'])
    hsbc_cfg = extractor.COMPANY_CONFIG['hsbc']
    assert hsbc_cfg['income']['expense_mode'] == 'label'
    santander_cfg = extractor.COMPANY_CONFIG['santander']
    assert any('profit' in label.lower() for label in santander_cfg['income']['net_income'])


def test_safe_division_handles_zero_and_none():
    assert extractor._safe_division(10.0, 2.0) == 5.0
    assert extractor._safe_division(None, 2.0) is None
    assert extractor._safe_division(10.0, 0.0) is None
