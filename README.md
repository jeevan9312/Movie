# 🎬 CineMatch - Hybrid Movie Recommendation System

A machine learning-powered **Hybrid Movie Recommendation System** that combines **Content-Based Filtering** and **Collaborative Filtering** to generate more relevant and personalized movie recommendations.

This project was built to demonstrate the **end-to-end machine learning workflow** — from data preprocessing and feature engineering to model building, hybrid recommendation logic, and deployment using **Streamlit**.

---

## 📌 Project Overview

Traditional recommendation systems often rely on a single technique, which can limit recommendation quality. To address this, **CineMatch** uses a **hybrid recommendation approach** by combining:

- **Content-Based Filtering** → Recommends movies similar in content (genres)
- **Collaborative Filtering** → Recommends based on user-item interaction patterns
- **Hybrid Weighted Blend** → Combines both approaches for better recommendation relevance

The application is deployed as an interactive web app using **Streamlit**, allowing users to explore movie recommendations in real time.

---

## 🚀 Live Demo

🔗 **Streamlit App:** [Add your deployed app link here]

---

## 🧠 Problem Statement

Movie recommendation systems are a classic and highly practical machine learning use case.  
However, relying on only one recommendation strategy can lead to limitations:

- **Content-based systems** may overfit to similar genres and fail to capture broader user behavior
- **Collaborative systems** may struggle with sparse data and cold-start scenarios

To solve this, this project implements a **hybrid recommender system** that leverages the strengths of both methods to improve the overall quality of recommendations.

---

## 🎯 Objectives

- Build a real-world **recommendation system**
- Understand and implement **multiple recommender techniques**
- Combine different recommendation strategies into a **hybrid system**
- Deploy the solution as a **user-facing ML application**
- Demonstrate an **end-to-end ML project pipeline**

---

## 📂 Dataset

This project uses the **MovieLens 100K Dataset**, a widely used benchmark dataset for recommendation systems.

### Dataset includes:
- **Users** → User IDs and demographics
- **Movies** → Movie titles and genres
- **Ratings** → User ratings for movies

### Why MovieLens 100K?
- Standard dataset for recommender system experimentation
- Balanced size for fast prototyping and deployment
- Suitable for collaborative filtering and hybrid modeling

---

## 🏗️ System Architecture

The project follows a modular ML pipeline:

1. **Data Collection**
   - Load MovieLens dataset
   - Parse users, movies, and ratings data

2. **Data Preprocessing**
   - Clean and merge relevant tables
   - Encode movie metadata
   - Prepare user-item interaction matrix

3. **Content-Based Filtering**
   - Extract genre information
   - Apply **TF-IDF Vectorization**
   - Compute **Cosine Similarity**

4. **Collaborative Filtering**
   - Build user-item rating matrix
   - Train **SVD / Matrix Factorization** model
   - Predict unseen user-movie preferences

5. **Hybrid Recommendation Engine**
   - Combine content-based scores and collaborative scores
   - Apply weighted blending logic
   - Rank final recommendations

6. **Deployment**
   - Build interactive UI using **Streamlit**
   - Deploy on **Streamlit Cloud**

---

## ⚙️ Recommendation Approaches

## 1️⃣ Content-Based Filtering

Content-based filtering recommends movies that are similar to the selected movie based on movie features.

### Technique used:
- **TF-IDF Vectorization** on movie genres
- **Cosine Similarity** to measure similarity between movies

### Workflow:
- Convert movie genre text into TF-IDF feature vectors
- Compute cosine similarity between all movies
- Recommend top-N most similar movies

### Strengths:
- Works well even without many user interactions
- Easy to explain and interpret
- Good for “similar movie” recommendations

### Limitations:
- Tends to recommend very similar content only
- May not capture broader user taste patterns

---

## 2️⃣ Collaborative Filtering

Collaborative filtering recommends movies based on user behavior and rating patterns.

### Technique used:
- **SVD (Singular Value Decomposition) / Matrix Factorization**

### Workflow:
- Create user-item rating matrix
- Factorize the matrix into latent factors
- Learn hidden relationships between users and movies
- Predict ratings for unseen items

### Strengths:
- Captures hidden user preferences
- Can recommend beyond obvious genre similarity
- Often more personalized than content-only systems

### Limitations:
- Suffers from **cold-start problem**
- Performance depends on sufficient user interaction data

---

## 3️⃣ Hybrid Recommendation System

The final recommendation engine combines both methods.

### Hybrid Logic:
- Generate recommendation scores from:
  - **Content-Based Filtering**
  - **Collaborative Filtering**
- Apply a **weighted blend** to combine both scores
- Rank movies based on the final hybrid score

### Why Hybrid?
A hybrid approach reduces the weaknesses of standalone methods by:
- improving recommendation diversity
- leveraging both content similarity and user behavior
- producing more robust results in practical scenarios

---

## 🧰 Tech Stack

### Languages & Libraries
- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Streamlit**

### ML Techniques
- **TF-IDF Vectorization**
- **Cosine Similarity**
- **SVD / Matrix Factorization**
- **Hybrid Weighted Recommendation Logic**

### Deployment
- **Streamlit Cloud**
