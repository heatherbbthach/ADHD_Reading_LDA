# Generated Outputs

The topic-modeling script (`src/02_topic_modeling.py`) generates coherence calculations, topic-modeling results, visualizations, and documents selected for qualitative interpretation.

Generated outputs include:

- **Coherence plots:** Visualizations of C_V coherence scores across candidate LDA models, including a plot highlighting the three-topic model.
- **Coherence tables:** `coherence_values_table.csv` and `coherence_values.csv`, containing calculated coherence scores.
- **Topic-modeling results:** Printed topic keywords and coherence scores for the final three-topic model.
- **pyLDAvis visualization:** An interactive HTML visualization of the final LDA model, along with its serialized visualization data.
- **Top documents per topic:** `top_documents_per_topic.txt`, containing the ten documents with the highest dominant-topic probabilities for each topic.

The original script's output filenames and locations have been retained to preserve the analytical workflow. Generated files are saved to the working directory (the repository root when the script is executed as instructed).

The `top_documents_per_topic.txt` file contains original Reddit text and should not be committed to this public repository to protect user privacy.

Coherence scores and visualizations are generated from the dataset used during execution and may differ from those reported in the published study.
