---
title: 'EnergyEfficientAI: A Python Library for Energy-Efficient Artificial Intelligence'
tags:
  - Python
  - machine learning
  - energy efficiency
  - power consumption
  - CPU monitoring
  - Green AI
  - Sustainability
authors:
  - name: Uzair Hassan
    email: uzair.zairi321@gmail.com 
    equal-contrib: true
    affiliation: "1"
  - name: Saif Ul Islam
    email: Saif.Islam@warwick.ac.uk
    equal-contrib: true
    affiliation: 2
  - name: Zia Ur Rehman
    email: zia.rahman@ist.edu.pk 
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Institute of Space Technology, Islamabad, Pakistan
   index: 1
 - name: WMG, University of Warwick, Coventry, UK
   index: 2
date: 14 October 2024
bibliography: paper.bib
---

# Summary

Energy-efficient computing has become a critical concern in the rapidly advancing field of artificial intelligence (AI), where the growing computational demands of machine learning (ML) algorithms pose significant environmental and economic challenges. **EnergyEfficientAI** is a Python library specifically developed to tackle this issue by monitoring and analyzing CPU and memory usage during ML model training, offering detailed insights into the energy footprint of various algorithms.

Designed with ease of integration in mind, EnergyEfficientAI provides a seamless interface for incorporating energy consumption tracking into existing ML workflows. It supports a broad range of models, from classical machine learning techniques to cutting-edge deep learning frameworks, making it a versatile tool for the AI community. The library enables users to visualize CPU utilization and generate comprehensive energy consumption reports, empowering researchers and engineers to optimize their models for accuracy and energy efficiency. These features contribute to the growing demand for sustainable AI practices by helping mitigate the environmental impact of intensive computational tasks.

# Statement of Need

As AI adoption accelerates across industries, the energy costs associated with training and deploying increasingly complex models have become a major concern. Despite the widespread availability of tools that monitor model performance—such as accuracy, precision, and loss—there remains a critical gap in tools that focus on energy consumption. **EnergyEfficientAI** addresses this void by allowing users to measure and analyze the energy consumption of their models, offering valuable insights into their computational efficiency.

This library is particularly essential for AI researchers and machine learning engineers working towards creating sustainable, energy-efficient AI systems. By offering a way to assess the energy footprint of different ML models, EnergyEfficientAI helps drive innovation in green AI and contributes to more environmentally conscious practices in AI development.

# Features

- Energy consumption calculation based on system power states.
- Real-time CPU and memory monitoring during model training.
- Visualizations of CPU usage during training.
- Compatibility with Scikit-learn, TensorFlow, and Keras models.

# Citations

The following works have been pivotal in the development and validation of **EnergyEfficientAI**:

1. **EnergyEfficientAI: A library to calculate power, energy, and training time of machine learning and deep learning algorithms** by Zia Ur Rehman, Uzair Hassan, and Saif Ul Islam.  
   The **EnergyEfficientAI** library has been made publicly available through the PyPI repository, offering a straightforward interface for researchers and engineers looking to optimize both model performance and energy efficiency.  
   [Link to Library](https://pypi.org/project/EnergyEfficientAI/#description)

2. **Towards green and sustainable artificial intelligence: quantifying the energy footprint of logistic regression and decision tree algorithms** by Uzair Hassan, Saif ul Islam, Syed Nasir Mehmood Shah, Zia Ur Rehman, and Jalil Boudjadar.  
   This research focuses on evaluating the energy consumption of classical machine learning algorithms, specifically logistic regression and decision trees. By quantifying their energy footprints, the paper highlights the environmental impact of these algorithms and underscores the need for energy-efficient models in the context of green AI. This work serves as a foundation for the development of tools like **EnergyEfficientAI**, which aim to promote sustainability in artificial intelligence research.  
   [Link to paper](https://digital-library.theiet.org/content/conferences/10.1049/icp.2024.2529)


# References
