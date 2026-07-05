"""
Tests for the routing functionality.
"""

import pytest
from nvidia_smartroute.routing.router import (
    RequestRouter,
    CapabilityAnalyzer,
    TaskType,
    ModelCapability,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        # Code words present but it's clearly generation, not maths.
        ("Write a Python function to calculate factorial", TaskType.CODE_GENERATION),
        # Word-boundary matching: "meaning" must not trigger maths ("mean").
        ("Summarize the meaning of this passage", TaskType.SUMMARIZATION),
        ("What is 2+2?", TaskType.MATHEMATICS),
        ("Explain why the sky is blue", TaskType.REASONING),
        ("Review this code and fix the bugs", TaskType.CODE_REVIEW),
        ("Translate this sentence to Japanese", TaskType.TRANSLATION),
        ("Write a short story about a dragon", TaskType.CREATIVE_WRITING),
        ("Complete the following code: def foo():", TaskType.CODE_COMPLETION),
        ("What does this code do?", TaskType.CODE_EXPLANATION),
        ("Hello, how are you today?", TaskType.CHAT),
    ],
)
def test_classifier_robust_cases(text, expected):
    analyzer = CapabilityAnalyzer()
    result = analyzer.classify([{"role": "user", "content": text}])
    assert result.task_type == expected
    assert 0.0 < result.confidence <= 1.0


def test_classifier_detects_vision_from_image_part():
    analyzer = CapabilityAnalyzer()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    assert analyzer.classify(messages).task_type == TaskType.VISION


def test_classifier_empty_messages_default_chat():
    analyzer = CapabilityAnalyzer()
    assert analyzer.classify([]).task_type == TaskType.CHAT


def test_task_type_detection():
    """Test that the capability analyzer correctly identifies task types."""
    router = RequestRouter()
    
    # Test code generation detection
    messages = [
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.CODE_GENERATION
    
    # Test code completion detection
    messages = [
        {"role": "user", "content": "Complete the following code: def hello_world(): "}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.CODE_COMPLETION
    
    # Test reasoning detection
    messages = [
        {"role": "user", "content": "Explain why the sky is blue"}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.REASONING
    
    # Test mathematics detection
    messages = [
        {"role": "user", "content": "Calculate the derivative of x^2 + 3x + 2"}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.MATHEMATICS
    
    # Test translation detection
    messages = [
        {"role": "user", "content": "Translate 'hello world' to Spanish"}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.TRANSLATION
    
    # Test summarization detection
    messages = [
        {"role": "user", "content": "Summarize the following article: [long article text]"}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.SUMMARIZATION
    
    # Test default to chat
    messages = [
        {"role": "user", "content": "Hello, how are you today?"}
    ]
    task_type = router.capability_analyzer.analyze_request(messages)
    assert task_type == TaskType.CHAT


def test_model_selection():
    """Test that the model selector chooses appropriate models for tasks."""
    router = RequestRouter()
    
    # Test code generation model selection
    decision = router.capability_analyzer.analyze_request([
        {"role": "user", "content": "Write a Python function to sort a list"}
    ])
    best_model = router.model_registry.select_best_model(decision)
    assert best_model is not None
    assert TaskType.CODE_GENERATION in best_model.supported_tasks
    
    # Test chat model selection
    decision = router.capability_analyzer.analyze_request([
        {"role": "user", "content": "What is the capital of France?"}
    ])
    best_model = router.model_registry.select_best_model(decision)
    assert best_model is not None
    assert TaskType.CHAT in best_model.supported_tasks
    
    # Test mathematics model selection
    decision = router.capability_analyzer.analyze_request([
        {"role": "user", "content": "Solve 2x + 5 = 15"}
    ])
    best_model = router.model_registry.select_best_model(decision)
    assert best_model is not None
    assert TaskType.MATHEMATICS in best_model.supported_tasks


def test_routing_decision():
    """Test that the router makes reasonable routing decisions."""
    import asyncio
    
    router = RequestRouter()
    
    async def route_request():
        return await router.route_request([
            {"role": "user", "content": "Write a Python function to calculate factorial"}
        ])
    
    # Run the async function
    decision = asyncio.run(route_request())
    
    assert decision.request_id is not None
    assert decision.task_type == TaskType.CODE_GENERATION
    assert decision.selected_model is not None
    assert decision.confidence > 0.0
    assert len(decision.reasoning) > 0
    assert TaskType.CODE_GENERATION in decision.selected_model.supported_tasks


def test_routing_stats():
    """Test that routing statistics are tracked correctly."""
    import asyncio
    
    router = RequestRouter()
    
    async def make_requests():
        # Make a few requests to generate stats
        await router.route_request([
            {"role": "user", "content": "Hello world"}
        ])
        await router.route_request([
            {"role": "user", "content": "Write a function to add two numbers"}
        ])
        await router.route_request([
            {"role": "user", "content": "What is 2+2?"}
        ])
    
    # Run the async function
    asyncio.run(make_requests())
    
    # Get stats
    stats = router.get_routing_stats()
    
    assert stats["total_decisions"] == 3
    assert "task_type_distribution" in stats
    assert "model_usage" in stats
    assert "average_confidence" in stats
    assert len(stats["recent_decisions"]) <= 3  # Should have at most 3 recent decisions
    
    # Check that we have the expected task types
    task_dist = stats["task_type_distribution"]
    assert TaskType.CHAT.value in task_dist
    assert TaskType.CODE_GENERATION.value in task_dist
    assert TaskType.MATHEMATICS.value in task_dist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])