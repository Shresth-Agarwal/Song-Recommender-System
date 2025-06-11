# Song Recommendation System

## Project Description

This project aims to build a song recommendation system that suggests songs to users based on their preferences. The system utilizes machine learning techniques, potentially incorporating natural language processing (NLP), to analyze song data and user behavior to provide personalized recommendations.

The system employs two main approaches:

1.  **Text-Based Recommendation:** This approach analyzes the lyrical content of songs using TF-IDF vectorization and Nearest Neighbors to find similar songs.
2.  **Feature-Based Recommendation:** This approach uses numerical features of songs (like danceability, energy, streams, etc.) and Nearest Neighbors to find songs with similar audio characteristics and popularity.

The system attempts text-based recommendations first and falls back to feature-based recommendations if a song is not found in the text-based dataset.

The project also includes a user-friendly web interface built with Streamlit.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. You can also run the core recommendation logic in Google Colab and potentially connect it to a deployed Streamlit app, or run both components locally.

### Prerequisites

To run this project locally, you will need:

*   Python 3.6 or higher
*   Jupyter Notebook or a Python environment that can run `.ipynb` files (for the notebook).
*   Streamlit (for the frontend).
*   Required Python libraries (listed in the `requirements.txt` section).

### Installing

1.  Clone the repository:
   ` git clone  `
2.  Navigate to the project directory:
   `cd song-recommendation-system`
3.  Install the required libraries. It's recommended to use a virtual environment:
   `pip install -r requirements.txt `

### Running the Streamlit Frontend

1.  Make sure you have the necessary datasets (`songdata.csv` and `popular_songs.csv`) in the same directory as your Streamlit application file (e.g., `app.py`).
2.  Run the Streamlit application from your terminal:
   `streamlit run app.py`
3.  (Replace `app.py` with the actual name of your Streamlit file).
3.  Your web browser should open with the Streamlit application.

### Running the Notebook in Google Colab

1.  Open the `Song_Recommendation_System.ipynb` notebook in Google Colab.
2.  Make sure you have the necessary datasets (`songdata.csv` and `popular_songs.csv`) uploaded to your Colab environment or connected from Google Drive. The notebook includes cells for uploading files.
3.  Run the cells sequentially to train the models.

## Usage

### Using the Streamlit Frontend

Open the Streamlit application in your web browser. You should see an interface where you can input a song title to get recommendations.

### Using the Core Recommendation Logic (in the Notebook)

Once the notebook is running, you can use the `recommender()` function in a code cell to get song recommendations.
   `recommender(song_title)`
Replace `"Song Title"` with the name of the song you want recommendations for. The system will output recommendations based on either the text-based or feature-based model.

## Datasets

This project uses the following datasets:

*   `songdata.csv`: Contains song lyrics and metadata for text-based recommendations.
*   `popular_songs.csv`: Contains popular song data with various audio features and streaming information for feature-based recommendations.

Make sure these files are accessible in the environment where you are running the notebook and/or the Streamlit application.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details. (If you have a LICENSE.txt file)

## Acknowledgments

*   Mention any datasets, libraries, or resources that were particularly helpful.
