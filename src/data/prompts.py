"""Shared prompt seeds for generation experiments.

All experiments (AR baseline, GazeDiffuse MDLM, GazeDiffuse LLaDA)
use the same 50 prompt seeds for fair comparison.
"""

from __future__ import annotations

# 50 diverse prompt seeds covering different topics and registers.
# Each prompt is 10-20 tokens to give models enough context.
PROMPT_SEEDS: list[str] = [
    "The discovery of a new species in the deep ocean",
    "Scientists have long debated whether artificial intelligence",
    "In the early morning hours, the city streets were",
    "The government announced a new policy that would",
    "Education plays a crucial role in shaping the",
    "The relationship between climate change and extreme weather",
    "A recent study published in Nature revealed that",
    "Technology has fundamentally changed the way people",
    "The history of space exploration began with",
    "Many researchers believe that the key to understanding",
    "The economic impact of the pandemic was felt",
    "In a small village on the coast of",
    "The development of renewable energy sources has",
    "Children who grow up reading books tend to",
    "The debate over social media's influence on",
    "Modern medicine has made remarkable advances in",
    "The ancient civilization left behind artifacts that",
    "Access to clean water remains a critical challenge",
    "The role of music in human culture has",
    "Recent advances in quantum computing suggest that",
    "The effects of sleep deprivation on cognitive",
    "In the field of neuroscience, researchers have",
    "The global food supply chain is increasingly",
    "Artists throughout history have used their work",
    "The importance of biodiversity for ecosystem health",
    "A growing body of evidence suggests that exercise",
    "The transition to electric vehicles is reshaping",
    "In many developing countries, access to healthcare",
    "The psychology of decision making reveals that",
    "Ocean temperatures have been rising steadily over",
    "The invention of the printing press revolutionized",
    "Urban planning must consider the needs of",
    "The human brain processes visual information through",
    "Advances in genetic engineering have opened up",
    "The cultural significance of food traditions in",
    "Studies show that bilingual individuals often have",
    "The architecture of ancient Rome continues to",
    "Climate scientists warn that if current trends",
    "The evolution of language has been shaped by",
    "Public health campaigns have successfully reduced the",
    "The mathematical foundations of machine learning rely",
    "In the aftermath of the earthquake, communities",
    "The philosophy of mind raises fundamental questions",
    "Conservation efforts in the Amazon rainforest have",
    "The rise of remote work has transformed how",
    "Archaeological evidence from the site indicates that",
    "The connection between gut bacteria and mental",
    "Innovations in materials science have led to",
    "The ethics of genetic modification continue to",
    "Throughout the twentieth century, advances in physics",
]


def get_prompts(n: int | None = None) -> list[str]:
    """Return prompt seeds for experiments.

    Args:
        n: Number of prompts to return. None returns all 50.

    Returns:
        List of prompt strings.
    """
    if n is None:
        return list(PROMPT_SEEDS)
    return list(PROMPT_SEEDS[:n])
