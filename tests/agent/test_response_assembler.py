"""Unit tests for ResponseAssembler covering progressive streaming transitions."""

from agent.response_assembler import ResponseAssembler


def test_transition_thought_to_text() -> None:
    """Verify transition from thought block to text item closes thought markdown."""
    assembler = ResponseAssembler()
    out1 = assembler.feed({"type": "thought", "content": "Pondering"})
    assert out1 == "> 💭 *Pondering"

    out2 = assembler.feed({"type": "text", "content": "Ready to act."})
    assert out2 == "> 💭 *Pondering*\n\nReady to act."
    assert assembler.finish() == out2


def test_transition_thought_to_status() -> None:
    """Verify transition from thought block to status item closes thought block."""
    assembler = ResponseAssembler()
    assembler.feed({"type": "thought", "content": "Analyzing"})
    out = assembler.feed({"type": "status", "content": "⚙️ Executing `move_to_pose`..."})
    assert out == "> 💭 *Analyzing*\n\n⚙️ Executing `move_to_pose`...\n"


def test_transition_text_to_status() -> None:
    """Verify transition from text to status appends newline before status."""
    assembler = ResponseAssembler()
    assembler.feed({"type": "text", "content": "Moving now."})
    out = assembler.feed({"type": "status", "content": "✅ Complete."})
    assert out == "Moving now.\n✅ Complete.\n"


def test_transition_status_to_text() -> None:
    """Verify transition from status to text preserves format without double newlines."""
    assembler = ResponseAssembler()
    assembler.feed({"type": "status", "content": "✅ Initialized."})
    out = assembler.feed({"type": "text", "content": "Continuing task."})
    assert out == "✅ Initialized.\nContinuing task."


def test_transition_error_after_thought() -> None:
    """Verify error item after thought closes thought block and formats error."""
    assembler = ResponseAssembler()
    assembler.feed({"type": "thought", "content": "Processing step."})
    out = assembler.feed({"type": "error", "content": "Hardware timeout"})
    assert out == "> 💭 *Processing step.*\n\n\n**Error:** Hardware timeout\n"


def test_consecutive_thoughts() -> None:
    """Verify consecutive thought items stream continuously within the same thought block."""
    assembler = ResponseAssembler()
    out1 = assembler.feed({"type": "thought", "content": "First thought."})
    assert out1 == "> 💭 *First thought."

    out2 = assembler.feed({"type": "thought", "content": " Second thought."})
    assert out2 == "> 💭 *First thought. Second thought."


def test_end_while_in_thought_block() -> None:
    """Verify finish() closes open thought block when stream terminates inside thoughts."""
    assembler = ResponseAssembler()
    assembler.feed({"type": "thought", "content": "Mid-thought termination"})
    final_text = assembler.finish()
    assert final_text == "> 💭 *Mid-thought termination*\n\n"


def test_full_multi_step_stream_sequence() -> None:
    """Verify multi-step sequence across thoughts, text, statuses, and error."""
    assembler = ResponseAssembler()
    events = [
        {"type": "thought", "content": "Thinking step 1..."},
        {"type": "thought", "content": " and step 2."},
        {"type": "text", "content": "I will check the joints."},
        {"type": "status", "content": "⚙️ Executing `get_joint_positions`..."},
        {"type": "status", "content": "✅ `get_joint_positions` completed."},
        {"type": "thought", "content": "Evaluating joints."},
        {"type": "error", "content": "Session error: Simulated connection failure"},
    ]
    chunks = [assembler.feed(ev) for ev in events]
    assert chunks == [
        "> 💭 *Thinking step 1...",
        "> 💭 *Thinking step 1... and step 2.",
        "> 💭 *Thinking step 1... and step 2.*\n\nI will check the joints.",
        (
            "> 💭 *Thinking step 1... and step 2.*\n\n"
            "I will check the joints.\n⚙️ Executing `get_joint_positions`...\n"
        ),
        (
            "> 💭 *Thinking step 1... and step 2.*\n\n"
            "I will check the joints.\n⚙️ Executing `get_joint_positions`...\n"
            "✅ `get_joint_positions` completed.\n"
        ),
        (
            "> 💭 *Thinking step 1... and step 2.*\n\n"
            "I will check the joints.\n⚙️ Executing `get_joint_positions`...\n"
            "✅ `get_joint_positions` completed.\n> 💭 *Evaluating joints."
        ),
        (
            "> 💭 *Thinking step 1... and step 2.*\n\n"
            "I will check the joints.\n⚙️ Executing `get_joint_positions`...\n"
            "✅ `get_joint_positions` completed.\n> 💭 *Evaluating joints.*\n\n\n"
            "**Error:** Session error: Simulated connection failure\n"
        ),
    ]
