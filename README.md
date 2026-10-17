# TB_DINO
TB-DINO: A Unified Foundation Model for Bi-temporal and Long-term Change Detection via Earth-scale Prior Knowledge Alignment
TB-DINO is a unified remote sensing change detection framework built upon the DINOv3 foundation model, designed to handle both conventional bi-temporal and long-term multi-temporal change detection tasks.

Unlike conventional change detection methods that are typically designed for a fixed number of temporal observations, TB-DINO introduces an Adaptive Temporal Interaction Module (ATIM) to model relationships among observations with varying temporal lengths. To further enhance change representations, TB-DINO incorporates complementary geospatial prior knowledge from AlphaEarth through a Similarity-guided Contrastive Alignment (SCA) module, which alleviates the representation gap between heterogeneous visual and geospatial features.

The DINOv3 backbone is efficiently adapted to remote sensing change detection using Low-Rank Adaptation (LoRA), enabling parameter-efficient transfer while preserving the general visual representations learned during large-scale pretraining.

Key Features Unified Framework: Supports both bi-temporal and long-term multi-temporal change detection. Foundation Model Adaptation: Adapts the DINOv3 foundation model to remote sensing change detection using parameter-efficient LoRA fine-tuning. Adaptive Temporal Modeling: Introduces ATIM to effectively model temporal observations with varying sequence lengths. Geospatial Prior Integration: Incorporates AlphaEarth representations to provide complementary geospatial prior knowledge. Cross-Modal Feature Alignment: Employs SCA to reduce the representation gap between visual features and heterogeneous geospatial representations before fusion. Efficient Training: Freezes the pretrained backbone and optimizes lightweight LoRA adapters and task-specific modules.

TB-DINO aims to provide a flexible and effective foundation-model-based solution for remote sensing change detection under different temporal observation settings.
