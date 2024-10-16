# testdnn
Deep Neural Network (DNN) Project: Predicting Structural Resistance of Steel Hollow Sections

This project investigates the application of simple deep neural network (DNN) architectures to predict experimental test results derived from an extensive numerical campaign on structural steel hollow sections. The dataset captures the structural resistance under different types of mechanical loading, along with input parameters such as geometry and material properties.

Despite the availability of numerous analytical formulas based on complex differential equations, this project aims to highlight an intriguing discovery: simple deep neural networks, when trained on high-quality data from numerical simulations, can sometimes outperform these traditional approaches in terms of prediction accuracy.

In this study, different neural network structures are tested, and the results in terms of prediction accuracy are compared to the expected outcomes. The project’s scope is to provide tools for testing similar case studies by varying the basic parameters that define a neural network. This allows for experimentation with different network architectures to evaluate their performance on comparable datasets.

For more in-depth analysis, including details on the dataset and the theoretical background of this work, please refer to:
https://onlinelibrary.wiley.com/doi/abs/10.1002/cepa.1398

An overview of some results is shown in the following plot. The x-axis represents the length of the simulated structure, while the y-axis shows the ratio of the expected result to the predicted resistance.

<details>
<summary>
  Results of the numerical simulations from the dataset compared to the DNN predictions along the specimen length.
</summary>


![image](https://github.com/user-attachments/assets/e01381d4-f20a-4e9d-b9fa-5ff32b621027)

</details>
