> **_PROJECT TASK:_**  We want to come up with an alphabet (a set of symbols) that have the following properties: they are mechanically distinguishable easily from scanned images (OCR), they are easy to write by hand.

Steps to perform OCR using the data:
1. clone this repository in your local environment.
``
https://github.com/amansingh2116/hindi_ocr_revised.git
``
2. From ``submit`` run ``knn.py`` to create the ``model.joblib`` file.
3. Add your test image to this directory and paste it's path in the image_path from ``ocr.py`` and run it to get the results. 

An overview of the Project codebase:
- ``submit`` contains the actual code to run the project, in which ``knn.py`` is to create the ``model.joblib`` and ``ocr.py`` is to actually perform ocr on a given image using ``pre.py`` and ``model.joblib``.
- ``sample images`` have the handwritten variations of the 64 characters as 30 x 30 px images. ``sample images copy``  contains the actual dataset where each image from ``sample image`` is used to design new dataset using data augmentation.
- TOOLS contains various partial codes that we used to create the data set like ``augmentation.py``, ``data_cleaning.py`` and ``get_data_segmentation.py``.

# Hindi Script Refinement & Improved OCR

The Hindi language, one of the most widely spoken languages globally, poses unique challenges for optical character recognition (OCR) due to its complex script. While Hindi is rich in cultural heritage and widely used in literature, administration, education, and everyday communication, its intricacies present obstacles for mechanical interpretation.

In this project, we address the challenges of OCR for Hindi by proposing a novel set of characters designed to enhance ease of writing and mechanical distinguishability. Recognizing the significance of simplicity in character design, we meticulously crafted a set of 60 symbols tailored to streamline both writing and OCR processes for Hindi.

Our approach aimed to bridge the gap between traditional script complexity and modern computational requirements. By introducing characters that are intuitive to write and easily discernible by machines, we sought to revolutionize the OCR landscape for Hindi language applications.

The culmination of our efforts resulted in the development of an OCR application specifically tailored for the newly designed Hindi characters. This application promises to significantly enhance the efficiency and accuracy of Hindi text recognition, opening doors to a myriad of applications across various domains.

Through this project, we aim to contribute to the advancement of Hindi language technology and facilitate greater accessibility and usability of Hindi content in the digital
age.

## Design Considerations:

**Simple shapes:** Utilize basic geometric shapes like lines, circles, squares, triangles, and combinations of these. Complex shapes or curves will be harder to distinguish in scans and write by hand.

**Distinctive Features:** Each symbol should have distinctive features that make it easy to differentiate from others. This could include variations in shape, size, orientation, or strokes.

**Consistency:** Maintain consistency in the design of the alphabet to ensure that each symbol is unique and recognizable. Consistent stroke width and proportions can aid in both writing and scanning.

**Limited Complexity:** Keep the overall complexity of the alphabet low to make it easier to learn and remember. This also helps with both writing and scanning accuracy.

**Legibility:** Prioritize legibility by avoiding symbols that are too similar or easily confused. Clear differentiation between characters is essential for both writing and scanning.

**Test and Refine:** Test the alphabet with various handwriting styles and scanning methods to ensure that it meets the desired criteria. Make adjustments as necessary based on feedback and usability testing.

**Size consistency:** Maintain a similar size for all symbols to avoid confusion during interpretation.

## Sample Alphabet Design

WE HAVE DESIGNED SOME (64) CHARACTERS AS ALPHABETS (attached as images/alpha1.jpeg) AS FIRST PROTOTYPE BEFORE ARRIVING AT THE FINAL SET OF ALPHABETS, AND PREPARED A TRAINING DATA SET FOR OCR HAVING AROUND 120 HANDWRITTEN VARIATIONS OF EACH CHARACTERS.

## Testing and Refinement:

**Scan various fonts and handwritten versions:** Test our symbols by scanning different fonts of the regular alphabet and various handwriting styles. Refine the symbols to ensure they remain distinguishable in all scenarios.

**Conduct user trials:** Have different people, including those with varying levels of handwriting skill, test the ease of writing the symbols by hand. Based on their feedback, adjust the symbols for better writability.

## Optical Character Recognition

To scan handwritten symbols and match them with your defined alphabet for detection, we can employ Optical Character Recognition (OCR) techniques. Here's a general approach to achieve this:

**Preprocessing:** Before performing OCR, preprocess the scanned images to enhance the quality and readability of handwritten symbols. This may include steps like binarization, noise removal, and resizing.

**Segmentation:** Divide the scanned image into individual symbols. This step is crucial for processing each symbol separately.

**Training Data Preparation:** Create a dataset containing examples of handwritten symbols along with their corresponding labels (i.e., the alphabet symbols). Ensure diversity in writing styles and variations.

**Feature Extraction:** Extract relevant features from each segmented symbol. Features could include stroke direction, curvature, and spatial relationships between different parts of the symbol.

**Training:** Train a machine learning model, such as a neural network, using the prepared dataset. Convolutional Neural Networks (CNNs) are commonly used for OCR tasks due to their effectiveness in learning spatial hierarchies of features.

**Classification:** Once the model is trained, use it to classify the handwritten symbols by predicting the corresponding alphabet symbols.

**Post-processing:** Apply post-processing techniques to refine the results and improve accuracy. This may involve techniques like voting among neighboring symbols or using language models to correct potential errors.

**Evaluation and Refinement:** Evaluate the performance of your OCR system using a separate validation dataset. Analyze errors and refine the system accordingly by adjusting parameters, improving preprocessing techniques, or augmenting training data.

By following these steps and leveraging machine learning techniques, we can develop an OCR system capable of scanning handwritten symbols and detecting them based on our defined alphabet. We should keep in mind that achieving high accuracy may require iterative refinement and optimization. Additionally, we can not use any existing OCR libraries or frameworks to streamline the development process.
