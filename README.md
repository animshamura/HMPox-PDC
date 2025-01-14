## Thesis Title:
**Bioinformatics and Deep Learning in Prognosis, Diagnosis, and Clinical Choice of Human Mpox Virus**



## Motivation:
- **Mpox as a Growing Threat**: Mpox is spreading globally, and quick, accurate diagnosis is essential to control outbreaks effectively.
- **Challenges with Traditional Methods**: Current diagnostic methods, like PCR, are slow, expensive, and often unavailable in low-resource settings.
- **Power of Deep Learning**: AI can analyze images and medical data to detect Mpox with high accuracy, making diagnosis faster and more reliable.
- **Role of Bioinformatics**: Bioinformatics helps study the virus genetic data, identify strains, and track mutations for better disease management.
- **Preparedness for Future Outbreaks**: Combining bioinformatics and AI can improve Mpox care and create a model for tackling other emerging diseases in the future.



## Objective:
- **Enhance Diagnostic Accuracy**: Use deep learning models to improve the precision and speed of Mpox detection.
- **Enable Early Prognosis**: Develop tools to predict disease severity and progression for better patient care.
- **Support Clinical Decision-Making**: Combine AI and bioinformatics to recommend personalized treatment options for Mpox patients.
- **Address Data Challenges**: Use synthetic data and transfer learning to overcome the limitations of small datasets.
- **Facilitate Genomic Understanding**: Apply bioinformatics to study virus mutations and strains to improve outbreak control and disease management.



## Literature Review:

### **Paper Title**: Monkey Pox Diagnosis with Interpretable Deep Learning

#### Working Processes:
- Data Collection and Augmentation: Collect lesion images and genomic data, and use augmentation (e.g., flipping, scaling) to improve diversity.
- Transfer Learning with Pre-trained Models: Use pre-trained models like VGG19 and MobileNetV2, fine-tuning them for Mpox diagnosis.
- Explainability with LIME: Employ LIME to highlight critical features like lesion size and shape, making AI predictions transparent.
- Model Evaluation: Evaluate models using metrics such as accuracy, precision, recall, and F1-score.
- Deployment in Clinical Tools: Integrate AI models into mobile apps or clinical decision systems for real-world use.

#### Limitations:
- Small Dataset Size: Limited data availability reduces the model’s accuracy and generalizability.
- Generalization Issues: Models may not perform well on diverse populations or varying imaging conditions.
- Lack of Clinical Validation: AI models are often untested in real-world clinical settings.
- High Computational Requirements: Training deep learning models demands significant hardware resources.
- Overfitting Risks: Small datasets lead to overfitting, reducing the model’s reliability.

#### Scope of Improvement:
- Expand and Diversify Datasets: Collect larger datasets and use GANs to generate synthetic data for better training.
- Improve Generalizability: Train models on multi-center datasets and adopt federated learning to ensure diverse performance.
- Conduct Clinical Validation: Test models in clinical environments to ensure their accuracy and usability.
- Develop Lightweight Models: Optimize models to run efficiently on mobile devices and low-resource systems.
- Integrate Imaging and Genomic Data: Combine lesion images with genomic data for comprehensive diagnostic solutions.



### **Paper Title**: PoxNet22: A Fine-Tuned Model for the Classification of Monkeypox Disease Using Transfer Learning

#### Working Processes:
- Data Preprocessing: Images were enhanced using contrast stretching, histogram equalization, and adaptive equalization to improve quality.
- Data Augmentation: Techniques like rotation, zooming, and feature-wise centering were applied to expand the dataset and reduce overfitting.
- Model Selection: Six pre-trained deep learning models were evaluated: DenseNet201, InceptionResNetV2, EfficientNetB7, InceptionV3, ResNet50, and VGG16. Performance metrics included accuracy, precision, recall, and loss.
- Proposed Model - PoxNet22: Based on InceptionV3 with fine-tuning for enhanced performance. Optimized with data augmentation, fine-tuned parameters, and ADAM optimizer.
- Evaluation and Performance: Achieved 100% accuracy, recall, and precision on the augmented dataset, demonstrating robust performance.

#### Limitations:
- Dataset Constraints: Limited availability of high-quality monkeypox skin lesion images.
- Generalization: Model performance might degrade with real-world, diverse datasets that include noise or varied lighting conditions.
- Dependency on Transfer Learning: Relies heavily on pre-trained models, which may not be fully optimized for monkeypox classification.
- Focus on Binary Classification: Primarily distinguishes between "monkeypox" and "others," lacking multi-class classification for other diseases with similar symptoms.

#### Scope of Improvement:
- Dataset Enhancement: Expand the dataset with diverse, real-world monkeypox lesion images. Collaborate with medical institutions to acquire more annotated samples.
- Incorporation of Multi-class Classification: Extend the model to differentiate between monkeypox, chickenpox, measles, and other skin conditions.
- Hybrid and Ensemble Models: Explore combining models or hybrid approaches for improved accuracy and robustness.
- Explainability and Interpretability: Use tools like Grad-CAM or SHAP to provide better insights into model decisions, aiding clinicians.
- Real-world Validation: Test the model in clinical settings to validate its effectiveness and reliability.
- Lightweight Implementation: Optimize the model for deployment on edge devices for faster, resource-efficient diagnosis.



### **Paper Title**: Utilizing Convolutional Neural Networks to Classify Monkeypox Skin Lesions

#### Working Processes:
- Data Preprocessing: Removed missing values and balanced the dataset using SMOTEEN for better representation and reduced noise.
- Feature Selection: Identified significant features such as fever, swollen lymph nodes, and oral lesions through correlation analysis.
- CNN Architecture: Implemented a convolutional neural network with specific layers optimized for monkeypox classification.
- Optimization with GWO: Fine-tuned CNN hyperparameters using the Grey Wolf Optimizer (GWO) for enhanced accuracy and performance.
- Evaluation Metrics: Assessed model performance using accuracy, precision, recall, F1-score, and AUC score, achieving 95.31% accuracy.

#### Limitations:
- Dataset Availability: Limited and non-representative monkeypox skin lesion images affected generalizability.
- Model Complexity: High computational resources and expertise required for CNN and GWO optimization.
- Overfitting Risk: Potential for overfitting due to limited and augmented datasets.
- Interpretability: Lack of explainability in CNN model decisions for clinical use.
- Real-world Validation: Insufficient testing in diverse and real-world clinical settings.

#### Scope of Improvement:
- Dataset Expansion: Collaborate with institutions to acquire a larger and more diverse dataset of annotated monkeypox images.
- Multi-class Classification: Extend the model to classify other similar skin conditions like chickenpox or herpes.
- Explainability: Implement explainable AI techniques like Grad-CAM to enhance model transparency.
- Efficiency: Optimize the CNN architecture for deployment in resource-constrained environments like mobile devices.
- Integration: Incorporate the model into telemedicine platforms for real-time diagnosis and improved healthcare accessibility.



### **Paper Title**: A deep-learning algorithm to classify skin lesions from mpox virus infection

#### Working Processes:
- Development of MPXV-CNN: Implementation of a deep convolutional neural network to classify mpox virus lesions with high sensitivity (0.91) and specificity (0.898).
- Dataset Compilation: Collection of 139,198 images, including 676 MPXV and 138,522 non-MPXV lesion images from diverse sources (public repositories, literature, social media, and clinical images).
- Performance Evaluation: Use of stratified fivefold cross-validation, external testing, and subgroup analyses to assess algorithm accuracy across different parameters (e.g., skin tone, body region).
- Integration into PoxApp: Development of a web-based app incorporating the MPXV-CNN to offer personalized patient guidance and risk stratification.
- SHAP Analysis: Utilization of SHapley Additive exPlanations to interpret CNN predictions by highlighting discriminative image regions.

#### Limitations:
- Dataset Bias: Limited MPXV image availability and potential biases due to image sourcing (e.g., extraordinary cases over typical ones).
- Skin Tone Representation: Underrepresentation of certain skin tones (e.g., Fitzpatrick types I and VI) in the dataset, impacting detection accuracy.
- Image Quality Issues: Algorithm performance decreases with low-quality images (e.g., low-light or blurry conditions).
- Differential Diagnoses Challenges: High false positive rates for specific conditions (e.g., orf, varicella, and molluscum contagiosum).
- Generalization: Limited evaluation under real-world conditions, including diverse populations and settings.

#### Scope of Improvement:
- Dataset Expansion: Acquire more diverse and representative MPXV images through multicenter trials and public contributions.
- Algorithm Refinement: Explore mobile-optimized architectures and uncertainty quantification to improve usability and reliability.
- Enhanced Training: Incorporate additional data for challenging differential diagnoses and skin tones with higher false positive rates.
- Clinical Trials: Conduct prospective studies to validate the app’s real-world effectiveness and compliance with recommendations.
- System Integration: Combine AI-based tools with expert clinical input and national early-warning systems for comprehensive outbreak management.



## Potential Solutions
- Integration of Imaging and Genomic Data: Combine lesion imaging data with genomic and proteomic data to create multi-modal diagnostic models, enhancing diagnostic precision by leveraging both visual and molecular biomarkers.
- Federated Learning for Multi-Center Dataset Training: Utilize federated learning to train models on diverse datasets from multiple institutions while ensuring data privacy and improving model generalizability.
- Dataset Expansion Through Synthetic Data Generation (GANs): Use generative adversarial networks (GANs) to generate synthetic images of Mpox lesions, addressing data scarcity and improving training diversity.
- Explainable AI (XAI) for Clinical Decision Support: Implement XAI tools (e.g., Grad-CAM, SHAP) to make deep learning model predictions interpretable, providing actionable insights for clinicians.
- Multi-Class Classification for Skin Lesion Diseases: Develop multi-class deep learning models to differentiate Mpox from other similar diseases (e.g., chickenpox, herpes) using combined imaging and molecular data.

## Gantt Chart 
<div align="center"> 
<img src="https://github.com/animshamura/Human-Mpox-Detection-and-Cure/blob/main/Diagrams/gantt.jpg" width="700" height="500"/> 
</div>

## Proposed Methodology 
<div align="center"> 
<img src="https://github.com/animshamura/Human-Mpox-Detection-and-Cure/blob/main/Diagrams/method.jpg" width="700" height="1000"/> 
</div>
