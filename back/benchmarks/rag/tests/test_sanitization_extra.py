import pytest
from unittest.mock import patch
from core.sanitization import (
    clean_answer_tokens,
    _fallback_parse_reasoning_response,
    extract_clean_answer
)

def test_clean_answer_tokens_edge_cases():
    assert clean_answer_tokens(None) == ""
    assert clean_answer_tokens("") == ""
    
    # Test think blocks masking and removal
    assert clean_answer_tokens("<think>private</think>hello") == "hello"
    assert clean_answer_tokens("<think>unclosed thought") == ""
    
    # Test source_id preservation
    assert clean_answer_tokens("<|source_id|> text") == "<|source_id|> text"
    
    # Test model-agnostic tag stripping
    assert clean_answer_tokens("<|some_tag|>text") == "text"
    assert clean_answer_tokens("<<inst>>text") == "text"
    assert clean_answer_tokens("[TECHNICAL_TAG]text") == "text"
    assert clean_answer_tokens("<s>text</s>") == "text"

def test_fallback_parse_reasoning_response_edge_cases():
    # Empty or non-string input
    assert _fallback_parse_reasoning_response(None) == ("UNKNOWN", "")
    assert _fallback_parse_reasoning_response(123) == ("UNKNOWN", "")
    
    # Unclosed status XML tag
    unclosed_status = "<|status_start|>ANSWERABLE"
    assert _fallback_parse_reasoning_response(unclosed_status)[0] == "ANSWERABLE"
    
    # Unclosed answer XML tag
    unclosed_answer = "<|answer_start|>my clean answer"
    assert _fallback_parse_reasoning_response(unclosed_answer)[1] == "my clean answer"
    
    # Status parsing with different keywords
    s1, _ = _fallback_parse_reasoning_response("### 4. _status\nNOT ANSWERABLE")
    assert s1 == "UNANSWERABLE"
    
    s2, _ = _fallback_parse_reasoning_response("### 4. _status\nSUFFICIENT")
    assert s2 == "ANSWERABLE"
    
    # No final answer headers
    s3, ans3 = _fallback_parse_reasoning_response("### 1. _analysis\nAnalysis\n### 3. _reasoning\nReasoning")
    assert "Analysis" not in ans3
    assert "Reasoning" not in ans3

def test_extract_clean_answer_edge_cases():
    assert extract_clean_answer(None) == ("UNKNOWN", "")
    assert extract_clean_answer(123) == ("UNKNOWN", "")
    
    # No final answer header, should return empty answer
    status, ans = extract_clean_answer("1. _analysis\nsome analysis text")
    assert status == "UNKNOWN"
    assert ans == ""

def test_fallback_parse_reasoning_response_more_xml():
    # Closed status XML tag (line 46)
    s, a = _fallback_parse_reasoning_response("<|status_start|>ANSWERABLE<|status_end|>")
    assert s == "ANSWERABLE"
    
    # Closed answer XML tag (line 71)
    s, a = _fallback_parse_reasoning_response("<|answer_start|>my answer<|answer_end|>")
    assert a == "my answer"
    
    # Markdown markers (line 90-91)
    s, a = _fallback_parse_reasoning_response("Final Answer: this is my response")
    assert a == "this is my response"

@patch("src.services.rag_service.parse_reasoning_response", side_effect=Exception("mock parse error"))
def test_extract_clean_answer_fallback_on_exception(mock_parse):
    # Reasoning headers but no final answer header -> empty answer (lines 136-137)
    status, ans = extract_clean_answer("1. _analysis\n2. _start\n3. _reasoning\n4. _status\nANSWERABLE")
    assert status == "ANSWERABLE"
    assert ans == ""
    
    # Normal response fallback (lines 146-147)
    status, ans = extract_clean_answer("<|answer_start|>fallback answer")
    assert status == "UNKNOWN"
    assert ans == "fallback answer"
