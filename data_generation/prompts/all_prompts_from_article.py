from .examples import examples

class PromptGenerator:
    """Generate prompts for creating question-answer pairs about Arab cultural content using a two-step process."""
    
    @staticmethod
    def build_prompt(question_type, metadata, num_pairs=1):
        """
        Build a complete prompt for a specific question type using a two-step process.
        
        Args:
            question_type (str): Type of question to generate
            metadata (dict): Information about the image (category, title, country, 
                            wikipedia_article, image_caption)
            num_pairs (int): Number of Q&A pairs to generate
            
        Returns:
            str: Complete prompt for generating augmented caption and Q&A pairs
        """
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"Unknown question type: {question_type}")
            
        qt = QUESTION_TYPES[question_type]
        
        # Step 1: Generate the augmented caption prompt with enhanced strictness
        step1_prompt = STEP1_TEMPLATE.format(
            focus_area=qt["focus_area"],
            task_description=qt["task_description"],
            question_type=question_type,
            specific_requirements=qt.get("caption_requirements", DEFAULT_CAPTION_REQUIREMENTS),
            **metadata
        )
        
        # Step 2: Generate the QA pairs prompt
        step2_prompt = STEP2_TEMPLATE.format(
            focus_area=qt["focus_area"],
            task_description=qt["task_description"],
            num_pairs=num_pairs,
            question_type=question_type,
            specific_requirements=qt.get("qa_requirements", DEFAULT_QA_REQUIREMENTS)
        )
        
        # Add example for this question type
        try:
            example_data = examples[question_type].copy()
            # Make sure we only pass expected keys to format
            expected_keys = {"ex_question", "ex_answer", "ex_augmented_caption"}
            filtered_example = {k: v for k, v in example_data.items() if k in expected_keys}
            # Fall back to defaults if keys aren't present
            if "ex_augmented_caption" not in filtered_example:
                filtered_example["ex_augmented_caption"] = "تظهر الصورة عناصر بصرية مفصلة وواضحة..."
            if "ex_question" not in filtered_example:
                filtered_example["ex_question"] = "ما هو العنصر الذي يظهر في الصورة؟"
            if "ex_answer" not in filtered_example:
                filtered_example["ex_answer"] = "العنصر الذي يظهر في الصورة هو عنصر ثقافي مهم."
                
            example = EXAMPLE_INTRO.format(**filtered_example)
        except (KeyError, AttributeError) as e:
            # Fallback if there's an issue with the example
            print(f"Warning: Issue with example for {question_type}: {e}")
            example = """## Example Output

```json
{
  "augmented_caption": "تظهر الصورة [وصف تفصيلي للعناصر المرئية]... [متبوعًا بمعلومات سياقية ذات صلة]",
  "generated_QAs": [
    {
      "question": "ما هو الغرض من العنصر الذي يظهر في الصورة؟",
      "answer": "العنصر الذي يظهر في الصورة هو [اسم العنصر المحدد]. وهو [إجابة مبنية على المعلومات المذكورة في الوصف المعزز]."
    }
  ]
}
```"""
        
        # Build the final combined prompt with stronger guardrails
        combined_prompt = COMBINED_TEMPLATE.format(
            step1_prompt=step1_prompt,
            step2_prompt=step2_prompt,
            example=example,
            num_pairs=num_pairs,
            question_type=question_type  # Explicitly pass question_type
        )
        
        return combined_prompt


# Templates
STEP1_TEMPLATE = """## Step 1: Generate an Augmented Caption

Your first task is to create a detailed, extended description of the image based on the provided image caption and Wikipedia article. This augmented caption should:

1. START with "تظهر الصورة" (The image shows) followed by a comprehensive description of what is VISUALLY present in the image, using the provided image caption as a foundation.
2. Expand on the visual elements by adding relevant contextual information from the provided sources.
3. Focus particularly on details that would support {focus_area} question-answer pairs.
4. Describe the image's visual elements in detail including:
   - Specific objects, people, settings, activities, and artifacts visible
   - Spatial relationships between elements
   - Notable colors, textures, and visual features
   - Any text or inscriptions visible
   - Any actions or events being depicted

### INFORMATION GUIDELINES:
1. Include information from the provided sources (category, title, country, image caption, Wikipedia article)
2. Present information as factual statements without attributing to the source
3. DO NOT use phrases like "according to the article," "as mentioned in the caption," or any reference to the sources
4. DO NOT mention Wikipedia, articles, captions, or sources in any way
5. Simply state the facts and information directly as established knowledge

### Question Type-Specific Requirements for {question_type}:
{specific_requirements}

### Input Information
- Cultural Category: {category}
- Article Title: {title}  
- Country: {country}
- Image Caption: {image_caption}
- Wikipedia Article: {wikipedia_article}

### REQUIRED STRUCTURE:
1. First paragraph: Begin with "تظهر الصورة" and describe what is VISUALLY present based on the image caption
2. Following paragraphs: Add relevant cultural context from the sources that directly relates to the visual elements
3. Final paragraph: Summarize elements specifically relevant to {question_type} questions

### CRITICAL VERIFICATION STEP:
Before finalizing your augmented caption, verify that:
- The caption has sufficient detail to support {question_type} question-answer pairs
- You have NOT included any reference to articles, captions, or sources
- You have presented all information directly as factual statements

If the provided sources do not provide SUFFICIENT information to create a meaningful augmented caption for {question_type} question-answer pairs, return a JSON error message:
```json
{{"error": "Insufficient context in the provided sources to generate an augmented caption for {question_type} question-answer pairs."}}
```
"""

STEP2_TEMPLATE = """## Step 2: Generate Question-Answer Pairs

Using ONLY the augmented caption you created in Step 1, now generate {num_pairs} different question-answer pairs that {task_description}, without explicitly naming it.

All questions and answers MUST be written in Modern Standard Arabic only.

### INFORMATION GUIDELINES:
1. Questions and answers MUST be based EXCLUSIVELY on information in the augmented caption
2. DO NOT introduce any new information not present in the augmented caption
3. DO NOT mention or reference any articles, captions, or sources
4. Present all information as direct factual statements without attribution
5. If the augmented caption lacks sufficient information for meaningful Q&A pairs, return an error

### Question Requirements:
1. Your question MUST reference something clearly described in the augmented caption
2. Your question MUST use one of these exact phrases to refer to the element:
   - "الذي يظهر في الصورة" (that appears in the image)
   - "كما يظهر في الصورة" (as shown in the image)
   - "الظاهر في الصورة" (the visible element in the image)
3. NEVER mention the specific name of the element in the question
4. Each question should require both visual identification AND cultural knowledge
5. NEVER include any terms that could hint at the exact name of the object, tradition, landmark, or feature

### Answer Requirements:
1. Base all answers EXCLUSIVELY on information in the augmented caption
2. NEVER add any details or context not in the augmented caption
3. Keep answers between 2-5 sentences in length
4. The answer MUST directly address and resolve the specific question being asked
5. Start your answer with a direct response to the question, then provide supporting details
6. DO explicitly name the object, tradition, or element in your answer
7. You SHOULD include the specific name of elements in the answer - unlike in the question
8. Use clear language and structured sentences that directly connect to the question
9. AVOID repeating the same information across multiple answers
10. Include the country/region name ONLY when it is relevant to the answer
11. NEVER mention or reference articles, captions, sources, Wikipedia, or any attribution phrases

### Question Type-Specific Requirements for {question_type}:
{specific_requirements}

### CRITICAL VERIFICATION STEP:
Before finalizing each Q&A pair, verify that:
- The answer contains information found ONLY in the augmented caption
- No new information has been introduced
- The question uses one of the required phrases
- The question doesn't name the specific element
- The answer DOES directly name the specific element
- The answer directly and clearly addresses the specific question being asked
- The answer follows the required length guidelines (2-5 sentences)
- NO reference is made to any sources such as articles or captions

For each question-answer pair, ask yourself:
- "Does this answer directly respond to what was asked in the question?"
- "Does the answer explicitly name the element that was kept hidden in the question?"
- "Would a reader understand how this answer relates to the question?"
- "Does the answer provide the specific information requested in the question?"
- "Have I avoided mentioning articles, captions, or sources in both the question and answer?"

If the augmented caption does not provide SUFFICIENT information to create meaningful question-answer pairs, return a JSON error message:
```json
{{"error": "Insufficient context in the augmented caption to generate meaningful question-answer pairs for this question type."}}
```
"""

COMBINED_TEMPLATE = """# Two-Step Process for Generating Question-Answer Pairs

{step1_prompt}

{step2_prompt}

{example}

## Final Output Format
Your final output must be a valid JSON object with the following structure:
```json
{{
  "augmented_caption": "The generated augmented caption based on the question-answer type.",
  "generated_QAs": [
    {{
      "question": "Your first question text in Modern Standard Arabic here",
      "answer": "Your first answer text in Modern Standard Arabic here"
    }},
    {{
      "question": "Your second question text in Modern Standard Arabic here",
      "answer": "Your second answer text in Modern Standard Arabic here"
    }}
    // Additional pairs as needed up to {num_pairs}...
  ]
}}
```

If there is insufficient context in the provided sources for either step, your output should be a JSON object with an error message:
```json
{{"error": "Specific error message explaining the lack of sufficient context."}}
```

## FINAL REMINDER - CRITICAL INSTRUCTIONS:
- The augmented caption MUST begin with "تظهر الصورة" followed by a detailed description of what is VISUALLY present
- ALL information must be based on the provided sources
- Present ALL information as direct factual statements
- DO NOT use phrases like "according to the article," "as mentioned," or similar attributions
- DO NOT reference Wikipedia, articles, captions, or any sources in the augmented caption, questions, or answers
- Questions should NOT mention the specific name of elements, but answers SHOULD explicitly name them
- Answers MUST directly respond to their corresponding questions and be between 2-5 sentences in length
- Each answer should begin with a sentence that directly addresses the question asked
- If you cannot generate meaningful content without inventing information, return an error message
"""

EXAMPLE_INTRO = """## Example Output

```json
{{
  "augmented_caption": "{ex_augmented_caption}",
  "generated_QAs": [
    {{
      "question": "{ex_question}",
      "answer": "{ex_answer}"
    }}
    // Additional pairs would be here if requesting more than one
  ]
}}
```
"""

# Default requirements for augmented captions
DEFAULT_CAPTION_REQUIREMENTS = """- Highlight aspects related to causes and effects, historical contexts, and cultural significance
- Include specific names, dates, and factual details that relate to the visual elements
- Structure information to progressively build from visual description to deeper cultural context
- Present all information as direct factual statements without attribution"""

# Default requirements for question-answer pairs
DEFAULT_QA_REQUIREMENTS = """- Each question must focus on a specific cultural element visible in the image
- Questions should NOT name the specific element, but use generic phrases
- Answers SHOULD explicitly name and identify the specific element
- Provide answers in 2-5 sentences that directly address the question asked
- Begin each answer with a clear statement that directly responds to the question
- Base questions strictly on the details found in the augmented caption
- Vary the cultural aspects you address
- Present all information as direct factual statements without attribution"""

# Question type definitions with focus area, task description, and requirements
QUESTION_TYPES = {
    "Cause and Effect": {
        "focus_area": "analyzes specific cause-effect relationships within cultural contexts",
        "task_description": "identifies the causes that led to a visible element's importance/significance and examines the resulting effects or impacts this element has within its cultural, religious, environmental, or societal framework",
        "caption_requirements": """- Explicitly identify cause-effect relationships
- Include clear statements about how elements came to be and their impacts
- Highlight historical events or factors that influenced the development of visible elements
- Include any consequences or outcomes related to the visual elements""",
        "qa_requirements": """- Clearly identify the cause (an element in the image) and the effect (a cultural or historical consequence)
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide answers in 2-5 sentences that directly explain the cause-effect relationship
- Begin each answer with a direct statement that addresses the specific causal relationship asked about
- Avoid generic cause-effect pairs; base them on the actual details found in the augmented caption
- Vary the cultural elements you address"""
    },
    "Chronological Sequence": {
        "focus_area": "about historical development and chronological ordering",
        "task_description": "requires identifying significant historical stages, periods, or evolutionary developments visible in the image and presenting them in proper temporal sequence. The answer should establish a clear chronological narrative that traces the subject's transformation over time, highlighting key dates, eras, or milestones from its origin/creation to its current state. Important transitions between time periods should be clearly articulated, with attention to distinctive features that characterize each phase of development",
        "caption_requirements": """- Identify and include all chronological markers (dates, periods, eras)
- Organize information in temporal sequence when presenting historical development
- Include any evolutionary stages or developmental phases for visible elements
- Highlight transitions between different time periods""",
        "qa_requirements": """- Each question must ask about the order or chronology of a culturally significant process, event, or tradition
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide an answer in 2-5 sentences that clearly presents the correct sequence
- Begin each answer with a direct statement that addresses the chronological question
- Reference visible or implied details from the augmented caption
- Use Modern Standard Arabic
- Vary the scenarios (preparation steps, ceremonial sequences, historical timeline, etc.)"""
    },
    "Comparative Analysis": {
        "focus_area": "compares or contrasts elements",
        "task_description": "asks to compare or contrast what's visible in the image with similar elements in other contexts, regions, or time periods",
        "caption_requirements": """- Include any comparisons or contrasts
- Note similarities and differences between elements
- Highlight regional variations or distinctions
- Include any information about how the visible elements compare to related cultural items""",
        "qa_requirements": """- Each question should compare at least two cultural aspects, either within the image or between the image and similar traditions in the same country
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide a clear, concise comparison in the answer (2-5 sentences)
- Begin each answer with a direct statement that addresses the comparative question
- Focus on specific elements mentioned in the augmented caption, not vague or overly broad topics
- Use Modern Standard Arabic
- Vary your comparisons (e.g., historical vs. modern, regional differences, or cross-cultural parallels)"""
    },
    "Modern Context": {
        "focus_area": "connects traditional Arab cultural elements to contemporary contexts",
        "task_description": "asks how what's visible in the image relates to contemporary Arab society, global influences, or modern practices",
        "caption_requirements": """- Include any information about contemporary relevance or modern usage
- Note any evolution from traditional to modern contexts
- Highlight current cultural significance or present-day applications
- Include any information about preservation efforts or modern adaptations""",
        "qa_requirements": """- Each question should ask how the traditional elements in the image can be integrated, preserved, or reimagined in modern version of the country
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide answers in 2-5 sentences, with culturally informed perspectives
- Begin each answer with a direct statement that addresses the modern context question
- Avoid broad statements; base questions on details explicitly mentioned in the augmented caption
- Address varied themes such as modernization, technological influence, or continuity of tradition"""
    },
    "General Question and Answer": {
        "focus_area": "straightforward factual",
        "task_description": """asks about what's visible in the image, its purpose, characteristics, or context by formulating:

- Two short questions (with answers),
- Two long question (with a longer, more detailed answer).""",
        "caption_requirements": """- Include a wide range of factual details about the visual elements
- Provide clear descriptions of purpose, function, and characteristics of visible items
- Include any cultural, historical, or regional context
- Organize information from general to specific, ensuring comprehensive coverage of facts""",
        "qa_requirements": """- Each question must be answerable by examining the augmented caption
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Answers must be 2-5 sentences, accurate, and directly address the question
- Begin each answer with a direct statement that responds to the specific question asked
- Use Modern Standard Arabic for all questions and answers
- Respect cultural sensitivities and avoid stereotypes
- Diversify your questions (observational, cultural attire, context setting, symbolism, T/F, multi-hop reasoning)
- Keep each question distinct from the others"""
    },
    "Hypothesis Formation": {
        "focus_area": "requires developing theories about cultural phenomena",
        "task_description": "asks the respondent to develop a theory explaining why what's visible in the image evolved or functions as it does within its cultural context",
        "caption_requirements": """- Include any explanations of purpose, function, or cultural significance
- Note any historical development or evolution described
- Include any information about cultural forces or factors that shaped the visible elements
- Highlight any descriptions of why certain features or characteristics exist""",
        "qa_requirements": """- Each question should encourage forming a hypothesis about the cultural role or significance of an element (e.g., a festival, artifact, or symbol)
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide an answer in 2-5 sentences, with reasoned justification grounded in cultural or historical context
- Begin each answer with a direct statement that addresses the hypothesis question
- Focus on details mentioned in the augmented caption
- Write in Modern Standard Arabic
- Vary your questions to address different cultural aspects, such as symbolism, origins, or evolutions"""
    },
    "Problem Solving": {
        "focus_area": "presents cultural challenges requiring solutions",
        "task_description": "presents a hypothetical challenge related to preserving, adapting, or utilizing what's visible in the image within a cultural context",
        "caption_requirements": """- Include any challenges, preservation efforts, or adaptations
- Note any historical or contemporary issues related to the visual elements
- Include any information about cultural evolution or adaptation
- Highlight any successful approaches or solutions""",
        "qa_requirements": """- Each question should pose a culturally grounded problem or challenge, such as how to preserve a traditional practice, adapt a craft to modern times, or address global influences
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide answers in 2-5 sentences, offering practical and culturally informed solutions
- Begin each answer with a direct statement that addresses the problem-solving question
- Keep the questions and answers relevant to the country, referencing the dimension category and the focus title
- Avoid generic problems; base them on specific elements mentioned in the augmented caption
- Use Modern Standard Arabic
- Vary the topics (e.g., educational initiatives, modernizing old crafts, balancing tradition and innovation)"""
    },
    "Origin Identification": {
        "focus_area": "about the origins of elements",
        "task_description": "asks about the historical, geographical, or cultural origins of what's visible in the image",
        "caption_requirements": """- Include any information about origins, history, or development
- Note geographical origins, cultural roots, or historical beginnings
- Include dates, locations, or cultural groups associated with origins
- Highlight any evolution or spread from original contexts""",
        "qa_requirements": """- Form each question around identifying the possible region, community, era, or cultural group related to a specific element from the image
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide concise answers (2-5 sentences) that contain accurate cultural or historical justifications
- Begin each answer with a direct statement that identifies the origin being asked about
- Avoid overly broad questions; focus on details from the augmented caption (e.g., attire, architecture, or symbolic motifs)
- Write the questions and answers in Modern Standard Arabic"""
    },
    "Perspective Shifting": {
        "focus_area": "examines Arab cultural elements from multiple viewpoints",
        "task_description": "asks to examine what's visible in the image from multiple viewpoints or frameworks (e.g., different social classes, generations, regions)",
        "caption_requirements": """- Include any information about different perspectives or viewpoints
- Note any variations in perception or significance across different groups
- Include any descriptions of how the visible elements are viewed by different communities
- Highlight any contrasting opinions or interpretations""",
        "qa_requirements": """- Each question should prompt analysis from at least two different perspectives (e.g., traditional vs. modern, rural vs. urban, different generations)
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide answers in 2-5 sentences that reflect these varied viewpoints
- Begin each answer with a direct statement that addresses the perspective question
- Base all perspectives on information from the augmented caption
- Use Modern Standard Arabic
- Vary the perspectives you include (e.g., historical figures, different social groups, regional variations)"""
    },
    "Role Playing": {
        "focus_area": "involves adopting specific roles related to Arab cultural elements",
        "task_description": "asks the respondent to adopt a specific role (e.g., artisan, historian, local resident) in relation to what's visible in the image",
        "caption_requirements": """- Include information about people, professions, or roles associated with the visible elements
- Note any descriptions of activities, responsibilities, or expertise related to cultural elements
- Include any details about how different people interact with or relate to the visible items
- Highlight any historical or contemporary roles"""
        ,
        "qa_requirements": """- Each question should specify a culturally relevant role (e.g., craftsperson, cultural minister, museum curator) and a task related to the visible element
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide answers in 2-5 sentences from that role's perspective
- Begin each answer with a direct statement that addresses the role-playing scenario
- Base the role and scenario on specific details from the augmented caption
- Use Modern Standard Arabic
- Vary the roles and scenarios across different aspects of cultural preservation, education, or promotion"""
    },
    "Scenario Completion": {
        "focus_area": "involves completing scenarios related to Arab cultural elements",
        "task_description": "presents an incomplete scenario involving what's visible in the image and asks for a logical conclusion or outcome",
        "caption_requirements": """- Include any sequential processes, procedures, or events
- Note any cause-and-effect relationships or outcomes
- Include any information about typical usage patterns or traditional practices
- Highlight any descriptions of how the visible elements are used or experienced""",
        "qa_requirements": """- Each question should present a culturally relevant scenario with missing information or an unresolved situation
- Questions should NOT name the specific element, but answers SHOULD explicitly name it
- Provide answers in 2-5 sentences that complete the scenario logically
- Begin each answer with a direct statement that addresses the scenario completion
- Base all scenarios on cultural elements explicitly mentioned in the augmented caption
- Use Modern Standard Arabic
- Vary the scenarios (e.g., festival preparations, craft demonstrations, preservation efforts, historical events)"""
    }
}

# Simple usage example
def get_prompt(question_type, metadata, num_pairs=1):
    """Convenience function to generate a prompt using the PromptGenerator class."""
    return PromptGenerator.build_prompt(question_type, metadata, num_pairs)