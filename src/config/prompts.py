"""
Prompt templates for GlobalHealthAtlas evaluator
"""

PROMPT_TEMPLATE = """
Your task is to evaluate a [Model Generated Response] (including its Chain of Thought) against a strictly verified [Standard Reference Answer] using a strict 1-10 scale.

# Cross-Lingual Analysis Protocol
**IMPORTANT:** The content in the Input Data (Question, Reference, Model Output) may be in **multiple languages (e.g., Chinese, English)**.
Regardless of the input language, your internal thought process, critique, reasoning, and the final JSON output must be written **strictly in ENGLISH**.

# Input Data
- **Domain:** {{domain}}
- **Task Label:** {{label}}
- **Question:** {{question}}
- **Standard Reference Answer:** {{answer}}
- **Standard Reference Reasoning:** {{complexCOT}}
- **Model Chain of Thought (COT):** {{llm_complexCOT}}
- **Model Final Response:** {{llm_answer}}

# Scoring Scale (1-10)
- **1:** Critical Failure. Completely wrong, dangerous, or irrelevant.
- **2:** Severe Failure. Contains no useful information or promotes harmful misinformation.
- **3:** Poor. Major errors evident.
- **4:** Weak. Significant missing info or confusion.
- **5:** Mediocre. Partially correct but lacks depth.
- **6:** Fair. Generally correct but lacks precision or detail.
- **7:** Good. Correct and professional.
- **8:** Very Good. Strong answer with only minor flaws.
- **9:** Excellent. Deep insight and high accuracy.
- **10:** Expert. Perfect alignment, expert terminology, and flawless logic.

---

# Dimensions & Detailed Rubrics

## 1. Accuracy
**Focus:** This dimension evaluates the absolute factual correctness of the model's response compared to the Standard Reference Answer.
**CRITICAL SCORING LOGIC:**

1. **IF Task Label == "Single-Choice":**
   **BINARY SCORING RULE APPLIES (0 or 10 ONLY):**
   - **CORRECT MATCH:** If the model's selected option matches the Standard Reference Answer exactly (e.g., Ref="B", Model="B"), you **MUST** assign an **INTEGER score of 10**.
   - **INCORRECT:** If the model selects a different option, multiple options, or fails to select an option, you **MUST** assign an **INTEGER score of 0**.
   - **DO NOT** assign any intermediate scores (e.g., 1-9) for Single-Choice tasks. It is pass (10) or fail (0).

2. **IF Task Label == "Question-Answer":**
   The model must accurately capture all specific key facts (e.g., numbers, entities).
   - **Rubric:**
     - **0 Points:** Major hallucinations; content is completely unrelated to the question.
     - **1 Point:** Wrong conclusions; key entities are fundamentally incorrect.
     - **2 Points:** Contains dangerous public health errors or severe factual contradictions.
     - **3 Points:** Captures the general topic but misses all specific values/data.
     - **4 Points:** Captures the general topic but misses the most critical specific data point.
     - **5 Points:** Correct values/conclusions but phrased ambiguously.
     - **6 Points:** Correct values/conclusions but lacks necessary precision.
     - **7 Points:** Factually correct but phrasing is slightly loose compared to Reference.
     - **8 Points:** Factually correct; minor stylistic differences only.
     - **9 Points:** Very strong accuracy, only trivial stylistic differences.
     - **10 Points:** PERFECT MATCH. All numbers, names, and conclusions match the Reference 100%.

## 2. Reasoning
**Focus:** This dimension assesses the logical validity and coherence of the model's step-by-step reasoning process (Chain of Thought/COT). It examines whether the derivation adheres to established public health guidelines, linking interventions to health outcomes (e.g., disease reduction) without logical gaps, circular reasoning, or causal fallacies.
**CRITICAL CONSTRAINT:** The evaluator must check for the presence of the `llm_complexCOT` field. If the input `{{llm_complexCOT}}` is empty, null, or missing, the model has failed to demonstrate its reasoning process, and you **MUST assign an INTEGER score of 0** to this dimension regardless of the final answer's correctness.

**Rubric:**
- **0 Points:** **No COT provided.**
- **1 Point:** Illogical steps; the reasoning makes no sense.
- **2 Points:** Circular reasoning or introduces dangerous public health inferences.
- **3 Points:** Large logical gaps; jumps from premises to conclusions without evidence.
- **4 Points:** Weak logic; connects ideas that do not strictly follow one another.
- **5 Points:** Steps are generally logical but superficial.
- **6 Points:** Explains "what" happened but fails to explain "why".
- **7 Points:** Clear cause-and-effect chain fitting epidemiological principles.
- **8 Points:** Strong deduction with only very minor gaps in explanation.
- **9 Points:** Flawless deduction; excellent use of logic.
- **10 Points:** Masterful. The COT demonstrates deep understanding of surveillance mechanics, linking technical changes directly to public health outcomes.

## 3. Completeness
**Focus:** This dimension measures the extent to which the model retrieves and includes all Key Information Points (KIPs) present in the Standard Reference Answer. It requires a holistic comparison to ensure no critical components—such as dual metrics or multi-part answers are omitted.

**Rubric:**
- **1 Point:** Irrelevant. The response fails to address the subject matter entirely.
- **2 Points:** Fragmentary. Contains only isolated keywords without context.
- **3 Points:** Peripheral. Discusses related topics but misses the specific core concept required.
- **4 Points:** Deficient. Identifies the core concept but provides no supporting details or evidence.
- **5 Points:** Partial. Covers the core concept but misses significant supporting details present in the Reference.
- **6 Points:** Adequate. Covers the core concept and some details, but omits critical context.
- **7 Points:** Substantial. Covers most key points but omits secondary nuances or implications.
- **8 Points:** Strong. Misses only one minor, non-critical detail compared to Reference.
- **9 Points:** Near Perfect. Comprehensive coverage with only trivial exclusions.
- **10 Points:** Exhaustive. Every single key concept, data point, and nuance in the Standard Reference is included.

## 4. Consensus Alignment
**Focus:** This dimension evaluates the model's adherence to established scientific consensus and authoritative guidelines from bodies such as the CDC, WHO, or ECDC. It scrutinizes the response for any claims that contradict accepted medical science, public health protocols, or standard operating procedures.

**Rubric:**
- **1 Point:** Contradicts established medical consensus (e.g., pseudoscientific claims).
- **2 Points:** Dangerous advice that violates safety protocols.
- **3 Point:** Plausible but not the standard accepted view.
- **4 Points:** Lacks authority; sounds like an opinion rather than a fact.
- **5 Points:** General agreement with consensus but vague.
- **6 Points:** Uses outdated guidelines or terminology.
- **7 Points:** Clearly reflects current public health standards and guidelines.
- **8 Points:** Strong alignment with standards; no contradictions.
- **9 Points:** Embodies the scientific consensus of the domain perfectly.
- **10 Points:** Perfectly embodies the scientific consensus, aligning strictly with implied CDC/WHO standards.

## 5. Terminology Norms
**Focus:** This dimension assesses the lexical precision and professional density of the language used. It demands the correct usage of domain-specific jargon rather than layperson or colloquial equivalents.

**Rubric:**
- **1 Point:** Uses layman/colloquial language (e.g., "bad virus").
- **2 Points:** Uses imprecise language (e.g., "strong test" instead of "high sensitivity").
- **3 Points:** Inconsistent use of terminology; mixes professional and casual terms frequently.
- **4 Points:** Attempts professional tone but fails often.
- **5 Points:** Uses correct terms generally, but lacks the density of an expert.
- **6 Points:** Uses correct terms but misses specific expert abbreviations.
- **7 Points:** Professional and consistent terminology usage throughout.
- **8 Points:** High degree of professional density; very few layman terms.
- **9 Points:** Academic precision. The language is indistinguishable from a top-tier research paper.
- **10 Points:** Flawless expert terminology usage.

## 6. Insightfulness
**Focus:** This dimension evaluates the depth of the mechanism explanation, distinguishing between mere fact retrieval and expert-level understanding. It asks whether the model explains the "Why" and "How" behind a phenomenon rather than just stating the result.

**Rubric:**
- **1 Point:** Mere repetition of the question.
- **2 Points:** Mere repetition of facts without explanation.
- **3 Points:** Explains *what* happened, but fails to explain the mechanism.
- **4 Points:** Attempts to explain mechanism but is unclear.
- **5 Points:** Basic explanation of the mechanism (e.g., "it works better because it's new").
- **6 Points:** Standard explanation of the mechanism without depth.
- **7 Points:** Deep insight; connects technical features to outcomes.
- **8 Points:** Connects technical features to epidemiological outcomes effectively.
- **9 Points:** Profound. Explains the underlying mechanism deeply.
- **10 Points:** Novel insight. Explains the underlying mechanism with expert depth.

---

# Output Instruction
1.  **TYPE ENFORCEMENT:** All scores must be strictly **INTEGERS** (e.g., 7, 8, 9). **DO NOT** output floats.
2.  **JSON STRUCTURE:** Return a flat JSON object where the keys are the exact dimension names. **Do NOT** include a `breakdown` wrapper, `total_score`, or `summary_verdict`.
3.  **REASONING:** The `description` field must explain *why* you assigned that specific score in **ENGLISH**, citing specific evidence from the model's response.

Return the result strictly in the following JSON format:

```json
{
  "Accuracy": {
    "score": <integer_0_to_10>,
    "description": "REASON: Explain the factual gap or exact match based on the Task Label."
  },
  "Reasoning": {
    "score": <integer_0_to_10>,
    "description": "REASON: Critique the logic flow. MUST state 'No COT provided' if score is 0."
  },
  "Completeness": {
    "score": <integer_0_to_10>,
    "description": "REASON: List specific missing points or confirm full coverage."
  },
  "Consensus Alignment": {
    "score": <integer_0_to_10>,
    "description": "REASON: Explain alignment or contradiction with scientific standards."
  },
  "Terminology Norms": {
    "score": <integer_0_to_10>,
    "description": "REASON: Cite examples of good or poor terminology usage."
  },
  "Insightfulness": {
    "score": <integer_0_to_10>,
    "description": "REASON: Evaluate the depth of the mechanism explanation."
  }
}
```
"""