# ML Model Serving via FastAPI and Docker  

## Introduction  

This repository is designed for deploying a Machine Learning model via a REST API built with FastAPI. The model is trained on a dataset sourced from the research conducted by OZKAN I.A., KOKLU M., and SARACOGLU R. (2021) in their paper *Classification of Pistachio Species Using Improved K-NN Classifier* (Progress in Nutrition, Vol. 23, N. 2, DOI: [10.23751/pn.v23i2.9686](https://doi.org/10.23751/pn.v23i2.9686)).  

The dataset used for training is publicly available at [this link](https://www.muratkoklu.com/datasets/).  

To ensure seamless deployment and reproducibility, the application is containerized with Docker, making it easy to run across different environments.  


## Getting Started 

To set up the repository properly, follow these steps:  

**1.** **Create the Data Directory**  
   - Before running the pipeline, create a `/data` folder in the project root.  
   - Inside `/data`, create two subdirectories:  
     - `/raw`: This will store the unprocessed dataset.  
     - `/processed`: The data will be split into **training and test sets** and saved here.
  

**2.** **Run the ML Pipeline**  
   - The `/src` folder contains modular components to execute the pipeline step by step:  
     - `load_data.py`: Ingests the data.  
     - `preprocess.py`: Stores the trained scaler and PCA models in `/models` for reuse in inference.  
     - `train_model.py`: Stores the trained model in `/models` for reuse in inference.  
     - `evaluate_model.py`: Computes metrics to validate the model's performance.  

**3.** **Deploy the API**  
   - The trained model is served via a REST API implemented in `main.py` with **FastAPI**.  
   - Follow these steps to deploy it using **Docker**:  

     ```bash
     # Build the Docker image
     docker build -t ml_model_api .
     
     # Run the container, exposing the API on port 8000
     docker run -p 8000:8000 ml_model_api  
     ```

   - Once the container is running, access the API at:  
     - **Swagger UI for interactive docs:** `localhost:8000/docs`  
     - **Health check endpoint:** `/health`  
     - **Prediction requests:** `/predict`  


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  
