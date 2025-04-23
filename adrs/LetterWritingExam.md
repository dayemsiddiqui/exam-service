# ADR: Letter Writing Exam Generation

**Status:** Proposed

**Context:**

- There is a need to generate "Letter Writing" style exams, specifically modeled after formats like the Telc B1 exam.
- These exams require generating a fictional letter/email stimulus (in German) and four specific points (also in German) that the examinee must address in their response.
- The stimulus can be either formal or informal.
- The generation process should align with existing exam generation workflows within the service.

**Decision:**

- We will create a new exam generation workflow specifically for Letter Writing exams.
- This workflow will leverage [mention specific core components/services/models used in existing workflows, e.g., a specific prompt generation service, template engine, etc.].
- A new prompt template will be designed to instruct the generation model (e.g., an LLM) to produce:
  - A German-language letter/email (formal or informal).
  - Four related points in German that must be addressed in a response.
- Relevant Pydantic models will be created to structure the output: one for the generated letter/email text and another for the list of task points.
- [Any other specific technical decisions? e.g., How will formality/informality be controlled? How will the 4 points relate to the letter content? Any specific data sources?]

**Consequences:**

- **Positive:**
  - Enables generation of a new, requested exam type.
  - Reuses existing infrastructure/patterns, potentially speeding up development and ensuring consistency.
- **Negative:**
  - Requires development effort to create the new workflow and prompt template.
  - May require fine-tuning or specific prompt engineering to ensure the quality and relevance of the generated letters and points.
  - Potential challenges in consistently generating varied but appropriate formal/informal tones.


### Sample Prompt

In Telc B1 exam the examinee receive a fictional letter/email (in german) that could be formal or informal that they need to respond to, the exam also contains four points (also in German) that the examinees need to address in their response to the letter/email

### Important

- Your task 4 points should also be in German instead of English

Given this format generate a mock writing exam.
