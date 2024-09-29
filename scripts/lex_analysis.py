import matplotlib.pyplot as plt
import matplotlib.patches as patches

languages = ["NR", "SS", "XH", "ZU"]
code2name = {
    "NR": "isiNdebele",
    "SS": "siSwati",
    "XH": "isiXhosa",
    "ZU": "isiZulu"
}

word_counts = {
    language: 0 for language in languages
}

morpheme_counts = {
    language: 0 for language in languages
}

max_morphemes = {
    language: 0 for language in languages
}

morpheme_bins = {
    language: [0]*10 for language in languages
}

tags_sets = {
    language: set() for language in languages
}

morphemes_sets = {
    language: set() for language in languages
}

word_morpheme_matches = {
    language: 0 for language in languages
}

morpheme_tag_mismatches = {
    language: 0 for language in languages
}

fig, axs = plt.subplots(len(languages), 2, figsize=(10, 10))

for i, language in enumerate(languages):
    lines = []
    with open(f"data/TRAIN/{language}_TRAIN_SURFACE.tsv") as f:
        lines_ = f.readlines()
        lines.extend(lines_)

    with open(f"data/TEST/{language}_TEST_SURFACE.tsv") as f:
        lines_ = f.readlines()
        lines.extend(lines_)

    # morpheme in 3rd column, word per line, tags in 4th column
    # check for morpheme count, word count, tags count
    # also check how many morphemes combine to form back the word
    for line in lines:
        morphemes = line.split("\t")[2].split("_")
        morpheme_counts[language] += len(morphemes)

        morpheme_bins[language][len(morphemes)-1] += 1

        if len(morphemes) > max_morphemes[language]:
            max_morphemes[language] = len(morphemes)
        word_counts[language] += 1
        
        for morpheme in morphemes:
            morphemes_sets[language].add(morpheme)

        tags_ = line.split("\t")[3].split("_")
        tags = []
        for tag in tags_:
            if "Dem" in tag: continue
            tags.append(tag)
        for tag in tags:
            tags_sets[language].add(tag)

        if "".join(morphemes).lower() == line.split("\t")[0].lower():
            word_morpheme_matches[language] += 1

        if len(morphemes) != len(tags):
            morpheme_tag_mismatches[language] += 1
            # print(morphemes)
            # print(tags)
            # exit()
        
    

    axs[i, 0].bar(range(1, 11), morpheme_bins[language])
    axs[i, 0].set_xlabel("Number of Morphemes")
    axs[i, 0].set_ylabel("Frequency")
    axs[i, 0].set_title(f"Morpheme Bins for {code2name[language]}")

    axs[i, 1].pie([word_counts[language] - word_morpheme_matches[language], word_morpheme_matches[language]], labels=["Non-Matches", "Surface-Canonical Matches"], autopct='%1.1f%%')
    axs[i, 1].set_title(f"Word Morpheme Matches for {code2name[language]}")

plt.tight_layout()

# Save the figure with a border
plt.savefig("combined_graphs.png", bbox_inches='tight', pad_inches=0.5)
plt.close()

# Load the saved image
image = plt.imread("combined_graphs.png")

# Create a new figure with a border
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)

# Add a border around the image
border = patches.Rectangle((0, 0), image.shape[1], image.shape[0], linewidth=10, edgecolor='black', facecolor='none')
ax.add_patch(border)

# Save the final image with the border
plt.savefig("combined_graphs_with_border.png")
plt.close()


for language in languages:
    print(f"{language} has {word_counts[language]} words and {morpheme_counts[language]} morphemes")

    # morphemes per word
    print(f"Average morphemes per word: {morpheme_counts[language] / word_counts[language]:.2f}")

    # max morphemes per word
    print(f"Max morphemes per word: {max_morphemes[language]}")
    print(morpheme_bins[language])

    # number of unique tags
    print(f"Number of unique tags: {len(tags_sets[language])}")
    print(f"Number of unique morphemes: {len(morphemes_sets[language])}")
    print(f"Tags per morpheme: {len(morphemes_sets[language])/len(tags_sets[language]):.2f}")

    # number of words that match the morphemes as a percentage
    print(f"Number of words that match the morphemes: {word_morpheme_matches[language]}")
    print(f"Percentage of words that match the morphemes: {word_morpheme_matches[language] / word_counts[language] * 100:.2f}%")
    print(f"Misalignments: {morpheme_tag_mismatches[language]}")
    print()

