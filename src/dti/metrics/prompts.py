from collections import UserList
from dataclasses import dataclass


@dataclass
class SubjectPromptSet:
    object_prompts: list[str]
    live_prompts: list[str]

    def __repr__(self) -> str:
        # If the lists are too long, truncate the output.
        if len(self.live_prompts) > 10 or len(self.object_prompts) > 10:
            return "\n".join(
                [
                    "SubjectPromptSet(",
                    f"  object_prompts ({len(self.object_prompts)}):",
                    *[f"    - {p}" for p in self.object_prompts[:5]],
                    "    ...",
                    *[f"    - {p}" for p in self.object_prompts[-5:]],
                    ")",
                    f"  live_prompts ({len(self.live_prompts)}):",
                    *[f"    - {p}" for p in self.live_prompts[:5]],
                    "    ...",
                    *[f"    - {p}" for p in self.live_prompts[-5:]],
                ]
            )
        return "\n".join(
            [
                "SubjectPromptSet(",
                f"  object_prompts ({len(self.object_prompts)}):",
                *[f"    - {p}" for p in self.object_prompts],
                ")",
                f"  live_prompts ({len(self.live_prompts)}):",
                *[f"    - {p}" for p in self.live_prompts],
            ]
        )


SIMPLE = SubjectPromptSet(
    object_prompts=[
        "a {} in the jungle",
        "a {} in the snow",
        "a {} on the beach",
        "a {} on a cobblestone street",
        "a {} on top of pink fabric",
        "a {} on top of a wooden floor",
        "a {} with a city in the background",
        "a {} with a mountain in the background",
        "a {} with a blue house in the background",
        "a {} on top of a purple rug in a forest",
        "a {} with a wheat field in the background",
        "a {} with a tree and autumn leaves in the background",
        "a {} with the Eiffel Tower in the background",
        "a {} floating on top of water",
        "a {} floating in an ocean of milk",
        "a {} on top of green grass with sunflowers around it",
        "a {} on top of a mirror",
        "a {} on top of the sidewalk in a crowded street",
        "a {} on top of a dirt road",
        "a {} on top of a white rug",
        "a red {}",
        "a purple {}",
        "a shiny {}",
        "a wet {}",
        "a cube shaped {}",
        # Following prompts are not in the DreamBooth paper, but are added for more diversity.
        "A {} with Japanese modern city street in the background",
        "A {} with a landscape from the Moon",
        "A {} among the skyscrapers in New York city",
        "A {} with a beautiful sunset",
        "A {} in a movie theater",
        "A {} in a luxurious interior living room",
        "A {} in a dream of a distant galaxy",
        "A photo of {} with ribbons",
        "A photo of golden {}",
        "A photo of {} made out of leathers",
        "A {} in a dense fog",
        "A {} under a starry night sky",
        "A {} beside a rushing waterfall",
        "A {} in a field of wildflowers",
        "A {} with a futuristic cityscape in the background",
        "A {} near the pyramids of Egypt",
    ],
    live_prompts=[
        "a {} in the jungle",
        "a {} in the snow",
        "a {} on the beach",
        "a {} on a cobblestone street",
        "a {} on top of pink fabric",
        "a {} on top of a wooden floor",
        "a {} with a city in the background",
        "a {} with a mountain in the background",
        "a {} with a blue house in the background",
        "a {} on top of a purple rug in a forest",
        "a {} wearing a red hat",
        "a {} wearing a santa hat",
        "a {} wearing a rainbow scarf",
        "a {} wearing a black top hat and a monocle",
        "a {} in a chef outfit",
        "a {} in a police outfit",
        "a {} wearing pink glasses",
        "a {} wearing a yellow shirt",
        "a {} in a purple wizard outfit",
        "a red {}",
        "a purple {}",
        "a wet {}",
        "a cube shaped {}",
        # Following prompts are not in the DreamBooth paper, but are added for more diversity.
        "a {} with Japanese modern city street in the background",
        "a {} with a landscape from the Moon",
        "a {} among the skyscrapers in New York city",
        "a {} with a beautiful sunset",
        "a {} in a movie theater",
        "a {} in a luxurious interior living room",
        "a {} in a dream of a distant galaxy",
        "a {} wearing a spacesuit, planting a flag on the moon",
        "a {} as a firefighter, extinguishing a fire in a skyscraper",
        "a {} in a wetsuit, surfing a giant wave in the ocean",
        "a {} in Victorian attire, attending a tea party in an elegant garden",
        "a {} in a snowsuit, skiing down a steep mountain",
        "a {} as an explorer, navigating through an icy Arctic landscape",
        "a {} in an elegant masquerade mask at a Venetian ball",
        "a {} gazing out over a misty mountain range",
        "a {} standing proudly in a field of sunflowers",
        "a {} exploring a vibrant coral reef underwater",
        "a {} relaxing in a hammock under palm trees at sunset",
    ],
)


COMPLEX_LV1 = SubjectPromptSet(
    object_prompts=[
        "Against a plain white studio backdrop, a {} lying flat",
        "On a marble floor, a {} resting",
        "On a glass tabletop, a {} placed at the center",
        "On a stone staircase outdoors, a {} resting on a step",
        "On a wooden pier above still water, a {} lying on the boards",
        "On patterned ceramic tiles, a {} placed neatly",
        "On snow-covered stairs, a {} resting",
        "On a sandy path through dunes, a {} lying on the sand",
        "On a sunlit balcony over a city street, a {} placed near the railing",
        "Next to an old brick wall, a {} resting on the ground",
        "On a stone plaza, a {} lying still",
        "Inside a glass greenhouse, a {} placed on the floor",
        "With rolling green hills behind, a {} lying on short grass",
        "With a city skyline in the background, a {} lying on the ground",
        "At the entrance of a museum hall, a {} placed on the floor",
        "Beside a calm lake, a {} lying on a rocky shore",
    ],
    live_prompts=[
        "In a bamboo forest, a {} wearing a small hat",
        "Under soft morning light in a quiet park, a {} sitting",
        "Against a plain white studio backdrop, a {} looking forward",
        "Beside a calm lake, a {} standing",
        "On a stone staircase outdoors, a {} wearing a scarf",
        "In a narrow alley with neon signs, a {} walking",
        "With a city skyline in the background, a {} wearing sunglasses",
        "At the entrance of a museum hall, a {} standing still",
        "On a sandy path through dunes, a {} trotting",
        "Inside a glass greenhouse, a {} looking to the side",
        "Near a lighthouse on the coast, a {} wearing a collar",
        "On a wooden pier above still water, a {} sitting calmly",
        "Next to an old brick wall, a {} wearing a ribbon",
        "At a high mountain pass with clouds below, a {} facing the camera",
        "On a sunlit balcony over a city street, a {} wearing a bowtie",
        "With rolling green hills behind, a {} walking toward the viewer",
    ],
)


COMPLEX_LV2 = SubjectPromptSet(
    object_prompts=[
        "On a rain-soaked city street at night with neon reflections, a {} lying on the sidewalk",
        "In a bamboo forest at sunrise with light fog, a {} resting on a stone path",
        "By the sea at dusk with waves and distant cliffs, a {} lying on wet sand",
        "On a stone bridge above a river with city lights, a {} placed on the ledge",
        "Inside a grand train station with sunbeams through the roof, a {} lying on the floor",
        "Under a moonlit sky with scattered clouds, a {} resting on a flat rock",
        "On a rooftop with a skyline and distant mountains, a {} placed near the edge",
        "Along a canal with reflections and old brick facades, a {} lying on the cobbles",
        "In a snowy village street with warm window lights at dusk, a {} resting on the snow",
        "At a coastal harbor with moored boats and a distant lighthouse, a {} lying on a wooden pier",
        "Inside a library hall with tall shelves and soft lamps, a {} placed on a reading table",
        "On a desert road with long shadows and drifting sand, a {} lying on the asphalt",
        "At the edge of a canyon with layered rock walls and sky above, a {} resting on a flat stone",
        "On a plaza with fountains and arches in the background, a {} placed on the ground",
        "In a foggy forest clearing with sun rays through trees, a {} lying on moss",
        "Under autumn trees with golden leaves on the ground, a {} resting on fallen leaves",
    ],
    live_prompts=[
        "On a rain-soaked city street at night with neon reflections, a {} wearing a hat",
        "In a bamboo forest at sunrise with light fog, a {} standing still",
        "By the sea at dusk with waves and distant cliffs, a {} wearing a scarf",
        "On a stone bridge above a river with city lights, a {} walking",
        "Inside a grand train station with sunbeams through the roof, a {} looking forward",
        "Under a moonlit sky with scattered clouds, a {} wearing a bowtie",
        "On a rooftop with a skyline and distant mountains, a {} facing the camera",
        "Along a canal with reflections and old brick facades, a {} walking",
        "In a snowy village street with warm window lights at dusk, a {} wearing a collar",
        "At a coastal harbor with moored boats and a distant lighthouse, a {} sitting",
        "Inside a library hall with tall shelves and soft lamps, a {} wearing glasses",
        "On a desert road with long shadows and drifting sand, a {} walking forward",
        "At the edge of a canyon with layered rock walls and sky above, a {} standing",
        "On a plaza with fountains and arches in the background, a {} wearing a ribbon",
        "In a foggy forest clearing with sun rays through trees, a {} sitting",
        "Under autumn trees with golden leaves on the ground, a {} wearing a scarf",
    ],
)


class StylePrompts(UserList):
    def __init__(self, styles: list[str]) -> None:
        super().__init__(styles)

    def __repr__(self) -> str:
        # If the list is too long, truncate the output
        if len(self.data) > 10:
            return "\n".join(
                [
                    "StylePrompts(",
                    f"  styles ({len(self.data)}):",
                    *[f"    - {p}" for p in self.data[:5]],
                    "    ...",
                    *[f"    - {p}" for p in self.data[-5:]],
                    ")",
                ]
            )
        return "\n".join(
            [
                "StylePrompts(",
                f"  styles ({len(self.data)}):",
                *[f"    - {p}" for p in self.data],
                ")",
            ]
        )


STYLE_SET = [
    "A toothbrush in {} style",
    "A water bottle in {} style",
    "A kitchen sink in {} style",
    "A laptop charger in {} style",
    "A coffee mug in {} style",
    "A computer in {} style",
    "A dog in {} style",
    "A light bulb in {} style",
    "A cat in {} style",
    "A hairbrush in {} style",
    "A desk lamp in {} style",
    "A garden hose in {} style",
    "A microwave oven in {} style",
    "A floor lamp in {} style",
    "A shower curtain in {} style",
    "A salt shaker in {} style",
    "A ceiling fan in {} style",
    "A electric kettle in {} style",
    "A grocery bag in {} style",
    "A laundry basket in {} style",
    "A remote control in {} style",
    "A houseplant in {} style",
    "An orange in {} style",
    "A chocolate cake in {} style",
    "A refrigerator in {} style",
    "A sofa in {} style",
    "An elephant in {} style",
    "A door knob in {} style",
    "A backpack in {} style",
    "A penguin in {} style",
    "A bathrobe in {} style",
    "A cereal bowl in {} style",
    "A wall clock in {} style",
    "A swimmer in {} style",
    "A tablecloth in {} style",
    "A light switch in {} style",
    "A cloud in {} style",
    "A flower vase in {} style",
    "A teddy bear in {} style",
    "A horse in {} style   ",
]


ALL_PROMPT_SETS = {
    "simple": SIMPLE,
    "complex_lv1": COMPLEX_LV1,
    "complex_lv2": COMPLEX_LV2,
    "style": StylePrompts(STYLE_SET),
}
