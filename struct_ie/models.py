from enum import Enum
from typing import List, Optional, Dict

import outlines
from outlines.samplers import greedy
from pydantic import BaseModel, constr, Field


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

    def __init__(self, model_name: str, entity_types_with_descriptions: Optional[Dict[str, Optional[str]]] = None,
                 device: str = "cpu"):
        """
        Initialize the extractor with a model and optional entity types.

        Args:
            model_name (str): The name of the transformer model to use.
            entity_types_with_descriptions (Optional[Dict[str, Optional[str]]]): A dictionary mapping entity types to their descriptions.
            device (str): The device to run the model on, defaults to "cpu".
        """
        self.model = outlines.models.transformers(model_name, device=device)
        self.entity_types_with_descriptions = entity_types_with_descriptions
        if entity_types_with_descriptions is not None:
            self.entity_types = list(entity_types_with_descriptions.keys())
            self.generator = self.create_generator(self.entity_types)
        else:
            self.entity_types = None
            self.generator = None

    def create_generator(self, entity_types: List[str]):
        """
        Creates a generator for extracting entities.

        Args:
            entity_types (List[str]): A list of entity types to extract.

        Returns:
            A generator function for extracting entities from text.
        """
        EntityType = Enum('EntityType', {etype.upper(): etype for etype in entity_types})

        class Entity(BaseModel):
            entity_span: constr(min_length=1, max_length=100)
            entity_type: EntityType

        class EntList(BaseModel):
            named_entities: Optional[List[Entity]] = Field(default_factory=list)

        generator = outlines.generate.json(self.model, EntList, sampler=greedy())
        return generator

    def adapt_text(self, text: str, prompt: Optional[str] = None, few_shot_examples: Optional[List[dict]] = None):
        """
        Adapts the input text with a prompt and examples for the model.

        Args:
            text (str): The input text from which to extract entities.
            prompt (Optional[str]): An optional prompt to guide the extraction process.
            few_shot_examples (Optional[List[dict]]): Optional few-shot examples to guide the model.

        Returns:
            str: The adapted text ready for the model to process.
        """
        if self.entity_types_with_descriptions is None:
            raise ValueError("Entity types with descriptions must be provided before extracting entities.")

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
        """
        Extracts entities from the given text.

        Args:
            text (str): The input text from which to extract entities.
            prompt (Optional[str]): An optional prompt to guide the extraction process.
            few_shot_examples (Optional[List[dict]]): Optional few-shot examples to guide the model.

        Returns:
            dict: The extracted entities in JSON format.
        """
        if self.entity_types_with_descriptions is None:
            raise ValueError("Entity types with descriptions must be provided before extracting entities.")
        adapted_text = self.adapt_text(text, prompt, few_shot_examples)
        output = self.generator(adapted_text)
        return output

    def update_entity_types(self, entity_types_with_descriptions: Dict[str, Optional[str]]):
        """
        Updates the entity types for extraction.

        Args:
            entity_types_with_descriptions (Dict[str, Optional[str]]): A dictionary mapping entity types to their descriptions.
        """
        self.entity_types_with_descriptions = entity_types_with_descriptions
        self.entity_types = list(entity_types_with_descriptions.keys())
        self.generator = self.create_generator(self.entity_types)
