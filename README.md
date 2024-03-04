# stat_sem2_project
 We want to come up with an alphabet (a set of symbols) that have the following properties: they are mechanically distinguishable easily from scanned images (OCR), they are easy to write by hand.

1. # Design Considerations:

Simple shapes: Utilize basic geometric shapes like lines, circles, squares, triangles, and combinations of these. Complex shapes or curves will be harder to distinguish in scans and write by hand.

Distinctive Features: Each symbol should have distinctive features that make it easy to differentiate from others. This could include variations in shape, size, orientation, or strokes.

Consistency: Maintain consistency in the design of the alphabet to ensure that each symbol is unique and recognizable. Consistent stroke width and proportions can aid in both writing and scanning.

Limited Complexity: Keep the overall complexity of the alphabet low to make it easier to learn and remember. This also helps with both writing and scanning accuracy.

Legibility: Prioritize legibility by avoiding symbols that are too similar or easily confused. Clear differentiation between characters is essential for both writing and scanning.

Test and Refine: Test the alphabet with various handwriting styles and scanning methods to ensure that it meets the desired criteria. Make adjustments as necessary based on feedback and usability testing.

Size consistency: Maintain a similar size for all symbols to avoid confusion during interpretation.

2. # Sample Alphabet Design

- A: ▲
- B: ◉
- C: ◔
- D: ◉▽
- E: ■
- F: ─│
- G: ◉⊜
- H: ║║
- I: ║
- J: ║⊂
- K: ╱
- L: ─┐
- M: ╲╱
- N: ╱╲
- O: ○
- P: ║⊙
- Q: ○⊥
- R: ╲╱
- S: ~
- T: ─
- U: ║⊂
- V: ╱║
- W: ╲╲
- X: ╲╱╱╲
- Y: ╱╲╲╱
- Z: ─╱

3. # Testing and Refinement:

Scan various fonts and handwritten versions: Test our symbols by scanning different fonts of the regular alphabet and various handwriting styles. Refine the symbols to ensure they remain distinguishable in all scenarios.

Conduct user trials: Have different people, including those with varying levels of handwriting skill, test the ease of writing the symbols by hand. Based on their feedback, adjust the symbols for better writability.

# What More ! 
- number of alphabets can be reduced, 
- what about small letters, punctuations and grammar of the language, 
- a single symbol that is easily distinguishable than others is what we need.



To scan handwritten symbols and match them with your defined alphabet for detection, we can employ Optical Character Recognition (OCR) techniques. Here's a general approach to achieve this:

Preprocessing: Before performing OCR, preprocess the scanned images to enhance the quality and readability of handwritten symbols. This may include steps like binarization, noise removal, and resizing.

Segmentation: Divide the scanned image into individual symbols. This step is crucial for processing each symbol separately.

Feature Extraction: Extract relevant features from each segmented symbol. Features could include stroke direction, curvature, and spatial relationships between different parts of the symbol.

Training Data Preparation: Create a dataset containing examples of handwritten symbols along with their corresponding labels (i.e., the alphabet symbols). Ensure diversity in writing styles and variations.

Training: Train a machine learning model, such as a neural network, using the prepared dataset. Convolutional Neural Networks (CNNs) are commonly used for OCR tasks due to their effectiveness in learning spatial hierarchies of features.

Classification: Once the model is trained, use it to classify the handwritten symbols by predicting the corresponding alphabet symbols.

Post-processing: Apply post-processing techniques to refine the results and improve accuracy. This may involve techniques like voting among neighboring symbols or using language models to correct potential errors.

Evaluation and Refinement: Evaluate the performance of your OCR system using a separate validation dataset. Analyze errors and refine the system accordingly by adjusting parameters, improving preprocessing techniques, or augmenting training data.

By following these steps and leveraging machine learning techniques, we can develop an OCR system capable of scanning handwritten symbols and detecting them based on our defined alphabet. We should keep in mind that achieving high accuracy may require iterative refinement and optimization. Additionally, we can not use any existing OCR libraries or frameworks to streamline the development process.