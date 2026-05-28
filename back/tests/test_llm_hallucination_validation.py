import pytest
from src.llm_engine.base import validate_no_hallucinations


def test_validate_no_hallucinations_valid_data():
    """Verify that normal non-repetitive lists, dicts, and texts pass without issues."""
    valid_data = {
        "authors": ["W. Hong", "W. Wang", "Q. Lv", "J. Xu"],
        "concepts": [
            {"name": "Machine Learning", "description": "A field of study that gives computers the ability to learn."},
            {"name": "Neural Network", "description": "A network of artificial neurons used in deep learning models."}
        ],
        "tags": ["AI", "DL", "NLP"],
        "summary": "This paper introduces a new approach to reinforcement learning by combining policy gradients with Q-learning."
    }
    
    # Should not raise any exceptions
    validate_no_hallucinations(valid_data)


def test_validate_no_hallucinations_list_length_threshold():
    """Verify ValueError is raised when a list exceeds the length threshold of 50 items."""
    long_list = ["item"] * 51
    with pytest.raises(ValueError, match="List length .* exceeds reasonable threshold"):
        validate_no_hallucinations(long_list)


def test_validate_no_hallucinations_repeating_cycles_in_list():
    """Verify ValueError is raised on consecutive repeating cycles (e.g. A, B, A, B, A, B)."""
    # 1. Cycle length 1 repeating 3 times
    cycle1 = ["Li", "Li", "Li"]
    with pytest.raises(ValueError, match="Repetitive loop detected in list"):
        validate_no_hallucinations(cycle1)
        
    # 2. Cycle length 2 repeating 3 times
    cycle2 = ["Li", "Zhang", "Li", "Zhang", "Li", "Zhang"]
    with pytest.raises(ValueError, match="Repetitive loop detected in list"):
        validate_no_hallucinations(cycle2)
        
    # 3. Repeating cycle embedded inside a longer valid list
    nested_cycle = ["W. Hong", "W. Wang", "Li", "Zhang", "Li", "Zhang", "Li", "Zhang", "J. Xu"]
    with pytest.raises(ValueError, match="Repetitive loop detected in list"):
        validate_no_hallucinations(nested_cycle)


def test_validate_no_hallucinations_low_uniqueness_ratio():
    """Verify ValueError is raised when uniqueness ratio in a large list is too low."""
    # List of size 12 with only 3 unique values (ratio = 3/12 = 0.25 < 0.4)
    low_uniqueness = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
    # Note: this also triggers the repeating cycle check.
    # Let's create one that does not trigger consecutive cycle of length <=4:
    # E.g. A, B, C, D, E, A, B, C, D, E, A, B (size 12, unique 5: ratio 5/12 = 0.41, let's make it lower)
    # A, B, C, D, E, F, G, A, B, C, D, E, F, G (size 14, unique 7: ratio 7/14 = 0.5)
    # Let's use list: ["A", "B", "C", "D", "E"] * 3 (size 15, unique 5, ratio = 5/15 = 0.33)
    # The cycle length here is 5 (which is greater than max_cycle_len=4 for list checks), so it won't trigger cycle check,
    # but it will trigger low uniqueness check!
    low_unique_list = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]
    with pytest.raises(ValueError, match="Low uniqueness ratio in list"):
        validate_no_hallucinations(low_unique_list)


def test_validate_no_hallucinations_text_loop():
    """Verify ValueError is raised when text contains repetitive word cycles."""
    # 1. 1-word loop repeating 4 times
    text1 = "This is a very very very very interesting paper."
    with pytest.raises(ValueError, match="Repetitive loop detected in text"):
        validate_no_hallucinations(text1)
        
    # 2. Phrase loop repeating 4 times
    text2 = "We show that the model is a model is a model is a model is a model."
    with pytest.raises(ValueError, match="Repetitive loop detected in text"):
        validate_no_hallucinations(text2)
        
    # 3. Safe short word repetitions (e.g. single letters or punctuation) should be ignored
    safe_text = "A A A A - - - - - - - -"
    validate_no_hallucinations(safe_text)


def test_validate_no_hallucinations_primitive_frequencies():
    """Verify ValueError is raised when any primitive element appears 3 or more times in a list."""
    # List where a single element appears 3 times but not in a simple sequence
    repetitive_list = ["W. Hong", "Li", "W. Wang", "Li", "J. Xu", "Li"]
    with pytest.raises(ValueError, match="repeated 3 times in list"):
        validate_no_hallucinations(repetitive_list)

    # Dictionary enclosing the list should also fail
    nested_data = {"authors": ["W. Hong", "Zhang", "W. Wang", "Zhang", "J. Xu", "Zhang"]}
    with pytest.raises(ValueError, match="repeated 3 times in list"):
        validate_no_hallucinations(nested_data)


def test_validate_no_hallucinations_longer_phrase_loops():
    """Verify ValueError is raised for longer phrase cycles (length 2..10) repeating 3 times."""
    # 6-word phrase repeating 3 times
    phrase_loop = (
        "we propose a new neural network we propose a new neural network we propose a new neural network"
    )
    with pytest.raises(ValueError, match="Repetitive loop detected in text"):
        validate_no_hallucinations(phrase_loop)

