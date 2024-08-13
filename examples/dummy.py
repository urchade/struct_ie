from struct_ie import EntityExtractor

entity_types_with_descriptions = {
    "Person": "Names of individuals",
    "Award": "Names of awards or honors",
    "Date": "Specific dates or date ranges, like days, months, or years",
    "Competition": "Names of competitions or tournaments",
    "Team": "Names of sports teams or organizations",
    "Other": "Any other named entities not covered by the other categories"
}

extractor = EntityExtractor("Qwen/Qwen2-0.5B-Instruct", entity_types_with_descriptions, device="cuda")

extractor.update_entity_types(entity_types_with_descriptions)

text = [
    "Cristiano Ronaldo won the Ballon d'Or. He was the top scorer in the UEFA Champions League in 2018.",
    "Lionel Messi who plays for Barcelona won the Golden Boot in 2019."
]

entities = extractor.extract_entities(text)
print(entities)
