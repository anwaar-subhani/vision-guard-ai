# VisionGuard-AI - Smart Surveillance System 

Overview

We built a smart surveillance system that detects different visual and audio anomalies from live CCTV footage. The system uses multiple deep learning models to identify events like fights, sudden falls, fire, crowd detection, gunshots, and human screams. I mainly worked on two modules: Gunshot Detection and Human Scream Detection. I’ll briefly explain both below.

Gunshot Detection Module

For gunshot detection, we started by collecting audio samples from Kaggle. Most of these recordings were clean (gunshots in silence), which wasn’t realistic for real-world scenarios.

So we tried to make the data closer to real life by mixing gunshot sounds with different background noises like traffic, crowd, and ambience. We also added similar sounding noises like firecrackers and other loud bursts so the model doesn’t just classify every loud sound as a gunshot.

After preprocessing, we had around 8,000 positive samples and 24,000 negative samples.

For the model, we used ResNet-101 with Mel-spectrograms. We converted audio into spectrogram images and trained the model on those images.

In the end, we achieved around 96% precision on unseen test data.

Human Scream Detection Module

For scream detection, instead of training a model from scratch, we used a pretrained model called YAMNet.

YAMNet is already trained on AudioSet, which has millions of audio clips and can recognize 521 different sound classes (like human voices, animals, environmental sounds, etc.).

The idea was to use YAMNet as a feature extractor. It converts audio into embeddings (basically a compressed representation of the sound that still keeps important features).

We collected around 3,200 audio samples, split evenly between scream and non-scream.

Our pipeline was pretty simple:
Audio → YAMNet → 1024-d embeddings → custom classifier

On top of those embeddings, we trained our own neural network for binary classification (scream vs non-scream).

We ended up getting around 83% accuracy on unseen test data.

Summary

Overall, the goal was to make a system that can work in more realistic surveillance conditions, not just clean datasets. Most of the effort went into making the data closer to real-world noise and choosing models that generalize well.
