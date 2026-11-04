"""Unit tests for deterministic NLP Lab components (no trained models needed)."""
from nlp_lab import codemix
from nlp_lab.data import map_label, parse_line
from nlp_lab.normalize import basic_normalize, romanized_fold
from nlp_lab.script import script_profile, token_script, tokenize


def test_script_identification():
    assert script_profile("ನಾನು ಚೆನ್ನಾಗಿದ್ದೀನಿ")["text_type"] == "native"
    assert script_profile("ನಾನು ಚೆನ್ನಾಗಿದ್ದೀನಿ")["dominant_script"] == "Kannada"
    assert script_profile("நான் நல்லா இருக்கேன்")["dominant_script"] == "Tamil"
    assert script_profile("സുഖമാണോ")["dominant_script"] == "Malayalam"
    assert script_profile("Yeno hengidiyaa")["text_type"] == "romanized"
    assert script_profile("ನಾನು today ತುಂಬಾ busy ಇದ್ದೀನಿ")["text_type"] == "mixed_script"
    assert script_profile("123 !!")["text_type"] == "other"


def test_tokenizer_keeps_indic_words_whole():
    toks = tokenize("ಇವತ್ತು class ತುಂಬಾ boring ide.")
    assert toks == ["ಇವತ್ತು", "class", "ತುಂಬಾ", "boring", "ide", "."]
    assert [token_script(t) for t in toks] == ["Kannada", "Latin", "Kannada", "Latin", "Latin", "Punct"]


def test_normalisation():
    # punctuation is tokenised one character at a time; elongation is reduced inside words
    assert basic_normalize("Superrrr movie!! https://x.y @fan 2020") == "superr movie ! ! <url> <user> <num>"
    assert romanized_fold("hengidiyaa") == romanized_fold("hengidiyaaa") == romanized_fold("hengidiya")
    assert romanized_fold("Pwoli") == "pvoli"


def test_codemix_indices():
    s = codemix.stats(["kn_rom", "en", "kn_rom", "kn_rom", "en", "univ"])
    assert s["language_tokens"] == 5 and s["language_independent_tokens"] == 1
    assert s["cmi"] == 40.0                     # 100 * (1 - 3/5)
    assert s["switch_points"] == 3 and s["i_index"] == 0.75
    assert abs(s["m_index"] - (1 - (0.36 + 0.16)) / (0.36 + 0.16)) < 1e-3
    mono = codemix.stats(["en", "en"])
    assert mono["cmi"] == 0 and not mono["is_code_mixed"]
    # script variants of one language are the same language
    assert not codemix.stats(["kn", "kn_rom"])["is_code_mixed"]


def test_dataset_line_parsing_and_labels():
    assert parse_line("Bro nin ello hogbutte bedu bro;Positive;\n") == ("Bro nin ello hogbutte bedu bro", "Positive")
    assert parse_line("a;b; c;Negative;") == ("a;b; c", "Negative")
    assert parse_line("Thala mass\tPositive\n") == ("Thala mass", "Positive")
    assert map_label("unknown_state") == "neutral"
    assert map_label("Mixed_feelings") is None
    assert map_label("not-kannada") is None
