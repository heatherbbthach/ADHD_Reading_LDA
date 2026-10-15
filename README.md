# Topic Modeling of ADHD Discourse on Reading Challenges

Code accompanying:

Thach, H., & Ridenour, L. (2025). Topic modeling of ADHD discourse
on reading challenges: Cognitive differences and systemic factors.
*Neurodiversity, 3*, 1–16.
https://doi.org/10.1177/27546330251395890

## Overview

This repository contains the Python code used for the computational
analysis reported in the article. The study used Latent Dirichlet
Allocation (LDA) to analyze 6,295 Reddit posts and comments concerning
ADHD and reading comprehension.

The repository includes code for:
- collecting Reddit data using PRAW;
- preprocessing text;
- evaluating candidate LDA models using coherence measures;
- fitting the final three-topic LDA model;
- generating topic visualizations; and
- identifying documents with high topic probabilities for qualitative
  interpretation.

## Requirements

- Python 3.11
- Dependencies listed in `requirements.txt`
- Reddit API credentials (required only for data collection)

## Installation

1. Clone the repository.
2. Create a virtual environment using Python 3.11:

   ```bash
   python3.11 -m venv .venv

## Data availability

Raw Reddit posts and comments are not distributed with this repository
to protect user privacy. The data-collection script is provided to
document the original collection procedure.
