# PyNAS

> **Disclaimer**  
> Code will be released upon acceptance of the paper.  
>  
> Developed by [ESA Φ-lab](https://philab.esa.int) and [Little Place Lab](https://www.littleplace.com/post/little-place-labs-announces-near-real-time-vessel-detection-classification-powered-by-sidus-space)

<table>
  <tr>
    <td>
      <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/659bd3d128676374f33915bd/gbnR7qI_4yzAXZRG7okov.jpeg" alt="ESA Φ-lab" height="100">
    </td>
    <td>
      <img src="https://media.licdn.com/dms/image/v2/D4D0BAQFtPj9y4P8iKA/company-logo_200_200/company-logo_200_200/0/1739206873060/little_place_labs_logo?e=2147483647&v=beta&t=nblD78izW1vko_P2mU9hqbIeB4fhVD7E6yivtuRH4AI" alt="Little Place Labs" height="100">
    </td>
  </tr>
</table>


---

## Overview

**PyNAS** is a modular Neural Architecture Search (NAS) framework developed by ESA Φ-lab and Little Place Lab, specifically designed for optimizing deployment on edge devices. It leverages advanced metaheuristic strategies—primarily Genetic Algorithms (GA)—to efficiently identify optimal deep learning architectures suitable for highly constrained computational environments.

---

## Key Features

- **Metaheuristic Optimisation**  
  Utilizes Genetic Algorithms (GA) to perform robust and flexible architecture optimization across diverse search spaces.

- **Model Architecture Selection**  
  Automates the discovery of neural architectures tailored to specific onboard applications and constraints.

- **Edge Device Compatibility**  
  Designs models explicitly for efficient inference on edge computing platforms, including IoT hardware and embedded systems.

- **Performance Metrics**  
  Supports evaluation based on predefined or custom metrics, such as latency, memory footprint, power consumption, and accuracy—critical for edge deployment scenarios.

---

## Customization

- **Constraint Definition**  
  Users can specify custom constraints (e.g., maximum parameter count, inference time budget) and objectives to guide the search process.

- **User-Friendly Interface**  
  Offers an intuitive and modular API, allowing seamless integration with existing ML pipelines and deployment workflows.

---

## Use Cases

- **IoT Applications**  
  Optimizes neural models for low-power IoT devices with limited computational and memory resources.

- **Remote Sensing**  
  Enhances efficiency and deployability of deep models on remote sensing platforms with onboard processing capabilities.

- **Autonomous Vehicles**  
  Enables real-time, low-latency neural inference on embedded systems in autonomous driving environments.

---
