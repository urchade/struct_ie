from enum import Enum
from pydantic import BaseModel, constr, Field
from typing import List, Optional, Dict
import outlines
from outlines.samplers import greedy


def transform_few_shot_examples(few_shot_examples):
    """Transforms few-shot examples for model input."""
    transformed_examples = []
    for example in few_shot_examples:
        transformed_output = {
            "named_entities": [
                {"entity_span": entity[0], "entity_type": entity[1]}
                for entity in example["output"]
            ]
        }
        transformed_examples.append({"input": example["input"], "output": str(transformed_output)})
    return transformed_examples


class EntityExtractor:
    """Entity extractor using a transformer model."""

    def __init__(self, model_name: str, entity_types_with_descriptions: Dict[str, Optional[str]], device: str = "cpu"):
        """Initialize the extractor with a model and entity types."""
        self.model = outlines.models.transformers(model_name, device=device)
        self.entity_types_with_descriptions = entity_types_with_descriptions
        self.entity_types = list(entity_types_with_descriptions.keys())
        self.generator = self.create_generator(self.entity_types)

    def create_generator(self, entity_types: List[str]):
        """Creates a generator for extracting entities."""
        EntityType = Enum('EntityType', {etype.upper(): etype for etype in entity_types})

        class Entity(BaseModel):
            entity_span: constr(min_length=1, max_length=100)
            entity_type: EntityType

        class EntList(BaseModel):
            named_entities: Optional[List[Entity]] = Field(default_factory=list)

        generator = outlines.generate.json(self.model, EntList, sampler=greedy())
        return generator

    def adapt_text(self, text: str, prompt: Optional[str] = None, few_shot_examples: Optional[List[dict]] = None):
        """Adapts the input text with a prompt and examples for the model."""
        if prompt is None:
            prompt = "Extract entities from this text."
        few_shot_examples_str = ""
        if few_shot_examples is not None:
            transformed_examples = transform_few_shot_examples(few_shot_examples)
            few_shot_examples_str = "\n\n".join(
                [f"Input text: '{example['input']}'\nJSON output: {example['output']}" for example in
                 transformed_examples]
            )
            few_shot_examples_str += "\n\n"
        entity_types_str = "\n".join(
            f"'{entity_type}': {description}" if description else f"'{entity_type}'"
            for entity_type, description in self.entity_types_with_descriptions.items()
        )
        adapted_text = (
            f"{prompt}\n"
            f"Allowed entity types:\n{entity_types_str}.\n\n"
            f"{few_shot_examples_str}"
            f"Input text: '{text}'\n"
            f"JSON output:"
        )
        return adapted_text

    def extract_entities(self, text: str, prompt: Optional[str] = None, few_shot_examples: Optional[List[dict]] = None):
        """Extracts entities from the given text."""
        adapted_text = self.adapt_text(text, prompt, few_shot_examples)
        output = self.generator(adapted_text)
        return output

    def update_entity_types(self, entity_types_with_descriptions: Dict[str, Optional[str]]):
        """Updates the entity types for extraction."""
        self.entity_types_with_descriptions = entity_types_with_descriptions
        self.entity_types = list(entity_types_with_descriptions.keys())
        self.generator = self.create_generator(self.entity_types)
