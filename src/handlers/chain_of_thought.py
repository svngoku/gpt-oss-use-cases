"""Chain of Thought (CoT) handler for improved reasoning.

Based on OpenAI Cookbook: https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThoughtFormat(Enum):
    """Supported Chain of Thought formats."""
    XML = "xml"
    MARKDOWN = "markdown"
    JSON = "json"
    PLAIN = "plain"


@dataclass
class ThoughtStep:
    """Represents a single step in chain of thought."""
    step_number: int
    thought: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ChainOfThoughtResponse:
    """Complete Chain of Thought response."""
    thoughts: List[ThoughtStep]
    final_answer: str
    raw_response: str
    format_used: ThoughtFormat


class ChainOfThoughtHandler:
    """Handler for Chain of Thought prompting and parsing."""
    
    # Default prompts for different CoT styles
    COT_PROMPTS = {
        "zero_shot": "Let's think step by step.",
        "few_shot": "Let me break this down step by step, similar to the examples:",
        "structured": """
Please provide your reasoning in the following format:
<thinking>
Step 1: [First thought]
Step 2: [Second thought]
...
</thinking>
<answer>[Final answer]</answer>
""",
        "self_consistency": "Provide multiple reasoning paths and select the most consistent answer.",
    }
    
    def __init__(self, format_type: ThoughtFormat = ThoughtFormat.XML):
        """Initialize Chain of Thought handler.
        
        Args:
            format_type: Format to use for parsing thoughts
        """
        self.format_type = format_type
        self.parsers = {
            ThoughtFormat.XML: self._parse_xml_thoughts,
            ThoughtFormat.MARKDOWN: self._parse_markdown_thoughts,
            ThoughtFormat.JSON: self._parse_json_thoughts,
            ThoughtFormat.PLAIN: self._parse_plain_thoughts,
        }
    
    def create_cot_prompt(
        self,
        question: str,
        cot_style: str = "zero_shot",
        examples: Optional[List[Dict[str, str]]] = None,
        custom_instruction: Optional[str] = None
    ) -> str:
        """Create a Chain of Thought prompt.
        
        Args:
            question: The question to answer
            cot_style: Style of CoT to use
            examples: Optional few-shot examples
            custom_instruction: Optional custom CoT instruction
            
        Returns:
            Formatted prompt with CoT instruction
        """
        base_prompt = self.COT_PROMPTS.get(cot_style, self.COT_PROMPTS["zero_shot"])
        
        if custom_instruction:
            base_prompt = custom_instruction
        
        if cot_style == "few_shot" and examples:
            prompt_parts = [base_prompt, "\n\nExamples:"]
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Question: {example.get('question', '')}")
                prompt_parts.append(f"Reasoning: {example.get('reasoning', '')}")
                prompt_parts.append(f"Answer: {example.get('answer', '')}")
            prompt_parts.append(f"\n\nNow, let's solve:\nQuestion: {question}")
        elif cot_style == "structured":
            prompt_parts = [
                f"Question: {question}",
                base_prompt
            ]
        else:
            prompt_parts = [
                f"Question: {question}",
                f"\n{base_prompt}"
            ]
        
        return "\n".join(prompt_parts)
    
    def parse_response(self, response: str) -> ChainOfThoughtResponse:
        """Parse a Chain of Thought response.
        
        Args:
            response: Raw response from model
            
        Returns:
            Parsed Chain of Thought response
        """
        parser = self.parsers.get(self.format_type, self._parse_plain_thoughts)
        thoughts, final_answer = parser(response)
        
        return ChainOfThoughtResponse(
            thoughts=thoughts,
            final_answer=final_answer,
            raw_response=response,
            format_used=self.format_type
        )
    
    def _parse_xml_thoughts(self, response: str) -> Tuple[List[ThoughtStep], str]:
        """Parse XML-formatted thoughts.
        
        Args:
            response: Response with XML tags
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        thoughts = []
        
        # Extract thinking section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        if thinking_match:
            thinking_text = thinking_match.group(1)
            
            # Parse individual steps
            step_pattern = r'Step\s+(\d+):\s*(.*?)(?=Step\s+\d+:|$)'
            steps = re.findall(step_pattern, thinking_text, re.DOTALL)
            
            for step_num, thought in steps:
                thoughts.append(ThoughtStep(
                    step_number=int(step_num),
                    thought=thought.strip()
                ))
        
        # Extract final answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else ""
        
        # Fallback if no structured format found
        if not thoughts and not final_answer:
            return self._parse_plain_thoughts(response)
        
        return thoughts, final_answer
    
    def _parse_markdown_thoughts(self, response: str) -> Tuple[List[ThoughtStep], str]:
        """Parse Markdown-formatted thoughts.
        
        Args:
            response: Response with Markdown formatting
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        thoughts = []
        lines = response.split('\n')
        
        current_step = 0
        in_reasoning = False
        current_thought = []
        final_answer = ""
        
        for line in lines:
            # Check for step headers
            step_match = re.match(r'^#+\s*Step\s+(\d+)', line, re.IGNORECASE)
            if step_match:
                # Save previous step if exists
                if current_thought:
                    thoughts.append(ThoughtStep(
                        step_number=current_step,
                        thought=' '.join(current_thought).strip()
                    ))
                    current_thought = []
                
                current_step = int(step_match.group(1))
                in_reasoning = True
            elif re.match(r'^#+\s*(Answer|Conclusion|Final)', line, re.IGNORECASE):
                in_reasoning = False
                if current_thought:
                    thoughts.append(ThoughtStep(
                        step_number=current_step,
                        thought=' '.join(current_thought).strip()
                    ))
                    current_thought = []
            elif in_reasoning and line.strip():
                current_thought.append(line.strip())
            elif not in_reasoning and line.strip():
                final_answer += line.strip() + ' '
        
        # Handle remaining content
        if current_thought:
            thoughts.append(ThoughtStep(
                step_number=current_step if current_step else len(thoughts) + 1,
                thought=' '.join(current_thought).strip()
            ))
        
        return thoughts, final_answer.strip()
    
    def _parse_json_thoughts(self, response: str) -> Tuple[List[ThoughtStep], str]:
        """Parse JSON-formatted thoughts.
        
        Args:
            response: Response with JSON structure
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        import json
        
        thoughts = []
        final_answer = ""
        
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            if isinstance(data, dict):
                # Extract thoughts
                if "thoughts" in data or "steps" in data or "reasoning" in data:
                    thought_data = data.get("thoughts") or data.get("steps") or data.get("reasoning", [])
                    
                    if isinstance(thought_data, list):
                        for i, thought in enumerate(thought_data, 1):
                            if isinstance(thought, dict):
                                thoughts.append(ThoughtStep(
                                    step_number=thought.get("step", i),
                                    thought=thought.get("thought", thought.get("content", str(thought))),
                                    reasoning=thought.get("reasoning"),
                                    confidence=thought.get("confidence")
                                ))
                            else:
                                thoughts.append(ThoughtStep(
                                    step_number=i,
                                    thought=str(thought)
                                ))
                
                # Extract answer
                final_answer = str(data.get("answer", data.get("conclusion", "")))
        
        except json.JSONDecodeError:
            # Fallback to plain parsing
            return self._parse_plain_thoughts(response)
        
        return thoughts, final_answer
    
    def _parse_plain_thoughts(self, response: str) -> Tuple[List[ThoughtStep], str]:
        """Parse plain text thoughts with basic patterns.
        
        Args:
            response: Plain text response
            
        Returns:
            Tuple of (thoughts, final_answer)
        """
        thoughts = []
        lines = response.split('\n')
        
        # Look for numbered steps or bullet points
        step_patterns = [
            r'^(\d+)[.)]\s*(.*)',  # 1. or 1)
            r'^Step\s+(\d+):?\s*(.*)',  # Step 1:
            r'^[-*]\s*(.*)',  # Bullet points
        ]
        
        current_step = 0
        final_answer_keywords = ['therefore', 'thus', 'conclusion', 'answer', 'finally', 'result']
        final_answer_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a step
            is_step = False
            for pattern in step_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    is_step = True
                    if len(match.groups()) == 2:
                        current_step = int(match.group(1))
                        thought = match.group(2)
                    else:
                        current_step += 1
                        thought = match.group(1) if match.groups() else line
                    
                    thoughts.append(ThoughtStep(
                        step_number=current_step,
                        thought=thought.strip()
                    ))
                    break
            
            # Check if this might be the final answer
            if not is_step:
                lower_line = line.lower()
                if any(keyword in lower_line for keyword in final_answer_keywords):
                    final_answer_lines.append(line)
                elif final_answer_lines:  # Continue collecting answer after keyword
                    final_answer_lines.append(line)
        
        # If no clear structure found, treat paragraphs as steps
        if not thoughts:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs[:-1], 1):  # All but last as steps
                thoughts.append(ThoughtStep(
                    step_number=i,
                    thought=para
                ))
            if paragraphs:
                final_answer_lines = [paragraphs[-1]]
        
        final_answer = ' '.join(final_answer_lines).strip()
        
        return thoughts, final_answer
    
    def validate_reasoning(
        self,
        response: ChainOfThoughtResponse,
        min_steps: int = 2,
        require_conclusion: bool = True
    ) -> Tuple[bool, List[str]]:
        """Validate the quality of Chain of Thought reasoning.
        
        Args:
            response: Parsed CoT response
            min_steps: Minimum number of reasoning steps required
            require_conclusion: Whether a final answer is required
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check minimum steps
        if len(response.thoughts) < min_steps:
            issues.append(f"Insufficient reasoning steps (found {len(response.thoughts)}, need {min_steps})")
        
        # Check for conclusion
        if require_conclusion and not response.final_answer:
            issues.append("No clear final answer or conclusion found")
        
        # Check for empty thoughts
        empty_steps = [s.step_number for s in response.thoughts if not s.thought.strip()]
        if empty_steps:
            issues.append(f"Empty reasoning in steps: {empty_steps}")
        
        # Check for repetitive thoughts
        thought_texts = [s.thought.lower() for s in response.thoughts]
        if len(thought_texts) != len(set(thought_texts)):
            issues.append("Repetitive reasoning detected")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def extract_confidence(self, response: ChainOfThoughtResponse) -> float:
        """Extract confidence level from reasoning.
        
        Args:
            response: Parsed CoT response
            
        Returns:
            Confidence score between 0 and 1
        """
        # Check if confidence is explicitly stated
        for thought in response.thoughts:
            if thought.confidence is not None:
                return thought.confidence
        
        # Look for confidence indicators in text
        confidence_text = response.raw_response.lower()
        
        high_confidence_indicators = ['certain', 'definitely', 'clearly', 'obviously', 'undoubtedly']
        medium_confidence_indicators = ['likely', 'probably', 'seems', 'appears', 'suggests']
        low_confidence_indicators = ['might', 'possibly', 'perhaps', 'uncertain', 'unclear']
        
        high_count = sum(1 for word in high_confidence_indicators if word in confidence_text)
        medium_count = sum(1 for word in medium_confidence_indicators if word in confidence_text)
        low_count = sum(1 for word in low_confidence_indicators if word in confidence_text)
        
        total = high_count + medium_count + low_count
        if total == 0:
            return 0.5  # Default neutral confidence
        
        confidence = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.2) / total
        return min(1.0, max(0.0, confidence))
