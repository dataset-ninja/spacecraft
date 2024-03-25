The authors established the **Spacecraft Dataset** tailored for spacecraft detection, instance segmentation, and part recognition. Their primary contribution lies in crafting this dataset using imagery of space stations and satellites, augmented with comprehensive annotations. These annotations encompass bounding boxes outlining spacecrafts and intricate masks delineating object parts, achieved through a combination of automated processes and manual refinement.

## Motivation

Space technologies play an indispensable role in numerous critical applications today, including communication, navigation, and meteorology, owing to advancements in computer vision and machine learning techniques. Over the past two decades, the space industry has witnessed a proliferation of machine learning-based applications. These applications range from self-navigation systems for collision avoidance and health monitoring of spacecraft to asteroid classification, among others. However, the burgeoning development of space technologies has led to an increased demand for space datasets. Most state-of-the-art models in space technology rely on deep learning methods, necessitating substantial annotated data for effective supervised training. Nevertheless, a major obstacle hindering the progress of these space technologies is the scarcity of publicly available datasets. This scarcity stems from the sensitive nature of space-related data and the exorbitant costs associated with obtaining space-borne images.

An essential technology in numerous space applications involves accurately localizing space objects using visual sensors, such as object detection and segmentation in images. Localization serves as a pivotal step toward vision-based pose estimation, crucial for tasks like docking, servicing, or debris removal. However, a significant challenge for space-based object detection and instance segmentation is the scarcity of accessible large datasets with thorough annotations. Since pixel-level masks are necessary as ground truth for training, constructing a segmentation dataset for any new domain can be exceedingly time-consuming. Moreover, the large number of parameters in modern neural networks often necessitate training on sizable datasets ranging from thousands to millions of samples, making the overall effort required to develop such a dataset prohibitively expensive.

Numerous research endeavors have aimed to mitigate the cost and manpower required for image mask annotation by exploring automation or semi-automation approaches. These efforts include interactive segmentation techniques, where human annotators utilize models to create initial masks and iteratively refine them through interaction. Additionally, weakly supervised annotation methods have emerged, wherein users provide minimal annotation information about the image masks. Another avenue of research seeks to eliminate the need for annotation altogether through self-supervised learning. However, self-supervised learning methods often exhibit inferior performance in detection and segmentation tasks compared to their supervised counterparts.

## Dataset description

The authors set out to advance space-based vision research by initiating the creation of a new publicly available dataset of space images. This endeavor marks the initial phase of a long-term objective to develop novel machine learning algorithms tailored for spacecraft object detection, segmentation, and part recognition tasks. Given the sensitive nature of space imagery, there is a scarcity of publicly accessible real satellite images. To address this gap, the authors curated a dataset comprising 3117 images of satellites and space stations sourced from both synthetic and real imagery and videos. Employing a bootstrap strategy during the annotation process, they aimed to minimize the manual effort required. Initially, the authors employed an interactive labeling method on a small scale. Subsequently, they utilized the labeled data to train a segmentation model, enabling the automatic generation of coarse labels for additional images. These coarse labels were then refined manually using the interactive tool. This iterative process continued, with more finely annotated images being produced and incorporated into the dataset until its completion.

## Initial data collection

In order for the dataset to effectively train practical models, a substantial amount of data collection is necessary. However, at this stage, only a limited amount of data was required. This smaller dataset served the purpose of testing feasible annotation methods that are user-friendly and capable of generating satisfactory masks.

<img src="https://github.com/dataset-ninja/spacecraft/assets/120389559/ef068c5c-8219-439b-a250-43129b0aa38b" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Process from data collection to image segmentation.</span>

##  Surveying annotation methods

Utilizing the initial dataset from the previous phase, the authors embarked on experiments employing various state-of-the-art models and tools for image segmentation. Given the intricate nature of spacecrafts, which necessitates specialized domain knowledge, the authors decided to segment spacecrafts into three distinct parts: *solar panel*, main *body*, and *antenna*, owing to their common observability and ease of identification. While exploring different segmentation methods, self-supervised and weakly supervised approaches emerged as appealing due to their minimal human interaction and labor requirements per image. However, these methods fell short of delivering satisfactory performance, primarily because satellites often feature numerous unorthodox and small components. Furthermore, refining the output predictions of self-supervised or weakly supervised approaches proved highly inefficient. Conversely, interactive segmentation methods offered notable advantages by enabling users to iteratively enhance the mask through manual inputs, aligning well with the objectives of this study. Following extensive experimentation, the authors selected Polygon-RNN++ as their model of choice. This model facilitates the decomposition of objects into small convex areas, allowing manual labeling of these areas based on their positions on the spacecraft. Additionally, the model permits users to freely adjust the mask at the pixel level by adding or removing key points.

<img src="https://github.com/dataset-ninja/spacecraft/assets/120389559/5e6b7e2a-c323-4cf9-91f5-956d5eb32157" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Example of an image collected and its annotated masks. Red mask: solar panel; blue mask: antenna; green mask: main body.</span>

## Data annotation

Initially, the authors manually labeled the first batch of the images. Subsequently, for all subsequent iterations, we capitalize on existing annotations to aid the labeling process. This is achieved by training state-of-the-art models to generate initial mask predictions for various spacecraft components. They employ the DeepLabV3 architecture, utilizing refined initial weights pretrained on ImageNet for the dataset. Three distinct models are trained to predict the full mask of *spacecraft*, the mask of *solar panel*, and the mask of *antenna*. Following prediction, the authors select accurate part predictions from each model for assembly and proceed to refine the final mask manually. As their annotated image repository expands, the trained models undergo further refinement using the latest dataset to enhance future predictions. Consequently, this iterative process reduces manual efforts and accelerates the labeling of new images.

<img src="https://github.com/dataset-ninja/spacecraft/assets/120389559/edd866b3-730c-4e54-9f64-2497c881957a" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Comparing masks before and after manual refinement. Left: input image. Centre: assembled masks from model predictions. Right: masks after manual refinement. The models here are trained on 1003 annotated images.</span>

Once the authors obtained a satisfactory number of masks through the bootstrap circuit, they began revisiting problematic images identified for further processing. These included images containing text that needed removal, similar images that were initially missed during filtering, and images deemed too challenging for identification even by human vision. Additionally, they revisited images for remasking if they were considered to be of low quality. Upon obtaining the mask labels, the authors computed tight bounding boxes around the spacecrafts in each image.

## Dataset statistic

The final dataset comprises 3117 images, all standardized to a resolution of 1280 × 720. Within these images, there are masks delineating 10350 parts belonging to 3667 *spacecraft*. The *spacecraft* objects vary widely in size, ranging from as small as 100 pixels to nearly occupying the entirety of the images. On average, each *spacecraft* occupies an area of 122318.68 pixels. Furthermore, specific parts such as the *antenna*, *solar panel*, and main *body* occupy areas averaging 22853.64, 75070.76, and 75090.92 pixels, respectively. To facilitate standardized benchmarking of segmentation methods, the authors partitioned the dataset into training and test subsets, comprising 2516 and 600 images, respectively.

<img src="https://github.com/dataset-ninja/spacecraft/assets/120389559/5c05db11-37c4-4d3f-8954-dc4570cd8bb0" alt="image" width="600">

<span style="font-size: smaller; font-style: italic;">Histograms of the spacecraft mask areas in the training and test set.</span>



