# `Struct-IE`: Structured Information Extraction with Large Language Models

`struc-ie` is a Python library for named entity extraction using a transformer-based model.

## Installation

You can install the `struct_ie` library from PyPI:

```bash
pip install struct_ie
```

## Usage

Here's an example of how to use the `EntityExtractor`:

### 1. Basic Usage

```python
from struct_ie import EntityExtractor

# Define the entity types with descriptions (optional)
entity_types_with_descriptions = {
    "Name": "Names of individuals like 'Jane Doe'",
    "Award": "Names of awards or honors such as the 'Nobel Prize' or the 'Pulitzer Prize'",
    "Date": None,
    "Competition": "Names of competitions or tournaments like the 'World Cup' or the 'Olympic Games'",
    "Team": None
}

# Initialize the EntityExtractor
extractor = EntityExtractor("Qwen/Qwen2-0.5B-Instruct", entity_types_with_descriptions, device="cpu")

# Example text for entity extraction
text = "Cristiano Ronaldo won the Ballon d'Or. He was the top scorer in the UEFA Champions League in 2018."

# Extract entities from the text
entities = extractor.extract_entities(text)
print(entities)
```

### 2. Usage with a Custom Prompt

```python
from struct_ie import EntityExtractor

# Define the entity types with descriptions (optional)
entity_types_with_descriptions = {
    "Name": "Names of individuals like 'Jean-Luc Picard' or 'Jane Doe'",
    "Award": "Names of awards or honors such as the 'Nobel Prize' or the 'Pulitzer Prize'",
    "Date": None,
    "Competition": "Names of competitions or tournaments like the 'World Cup' or the 'Olympic Games'",
    "Team": "Names of sports teams or organizations like 'Manchester United' or 'FC Barcelona'"
}

# Initialize the EntityExtractor
extractor = EntityExtractor("Qwen/Qwen2-0.5B-Instruct", entity_types_with_descriptions, device="cpu")

# Example text for entity extraction
text = "Cristiano Ronaldo won the Ballon d'Or. He was the top scorer in the UEFA Champions League in 2018."

# Custom prompt for entity extraction
prompt = "You are an expert on Named Entity Recognition. Extract entities from this text."

# Extract entities from the text using a custom prompt
entities = extractor.extract_entities(text, prompt=prompt)
print(entities)
```

### 3. Usage with Few-shot Examples

```python
from struct_ie import EntityExtractor

# Define the entity types with descriptions (optional)
entity_types_with_descriptions = {
    "Name": "Names of individuals like 'Jean-Luc Picard' or 'Jane Doe'",
    "Award": "Names of awards or honors such as the 'Nobel Prize' or the 'Pulitzer Prize'",
    "Date": None,
    "Competition": "Names of competitions or tournaments like the 'World Cup' or the 'Olympic Games'",
    "Team": "Names of sports teams or organizations like 'Manchester United' or 'FC Barcelona'"
}

# Initialize the EntityExtractor
extractor = EntityExtractor("Qwen/Qwen2-0.5B-Instruct", entity_types_with_descriptions, device="cpu")

# Example text for entity extraction
text = "Cristiano Ronaldo won the Ballon d'Or. He was the top scorer in the UEFA Champions League in 2018."

# Few-shot examples for improved entity extraction
demonstrations = [
    {"input": "Lionel Messi won the Ballon d'Or 7 times.", "output": [("Messi", "Name"), ("Ballon d'Or", "Award")]}
]

# Extract entities from the text using few-shot examples
entities = extractor.extract_entities(text, few_shot_examples=demonstrations)
print(entities)
```

## License

This project is licensed under the Apache-2.0.