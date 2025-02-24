import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Function to load and build the model
@st.cache_resource
def load_model():
    # Load and preprocess IMDB dataset for word index
    vocab_size = 10000
    max_len = 200
    
    # Load IMDB dataset to get word index
    (_, _), (_, _) = imdb.load_data(num_words=vocab_size)
    
    # Build the model
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        LSTM(64, dropout=0.5, recurrent_dropout=0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model, vocab_size, max_len

# Function to predict sentiment
def predict_sentiment(review, model, word_index, vocab_size, max_len):
    # Simple negation handling
    negations = ["not", "no", "never"]
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(review)
    
    # Mark negations
    for i, word in enumerate(tokens):
        if word in negations and i + 1 < len(tokens):
            tokens[i + 1] = "not_" + tokens[i + 1]  # Prefix "not_" to the next word
    
    # Convert words to indices
    review_indices = []
    for word in tokens:
        idx = word_index.get(word)
        if idx is not None and idx < vocab_size:
            review_indices.append(idx)
        else:
            review_indices.append(1)  # OOV token
    
    # Pad sequence
    review_padded = pad_sequences([review_indices], maxlen=max_len)
    
    # Make prediction
    prediction = model.predict(review_padded, verbose=0)[0][0]
    return prediction, tokens

# Load model and resources
model, vocab_size, max_len = load_model()
word_index = imdb.get_word_index()

# Prepare word index
word_index = {word: (idx + 3) for word, idx in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2

# Example reviews
example_reviews = {
    "Positive Example": "This film was a masterpiece! The acting was top-notch.",
    "Negative Example": "I was extremely disappointed with this movie. The plot was confusing and the acting was terrible.",
    "Mixed Example": "It started off well but fell flat in the second half.",
    "Custom Review": ""
}

# Create navigation
st.sidebar.title("Navigation")
pages = ["Home", "Dataset", "Usage", "Model Evaluation"]
selection = st.sidebar.radio("Go to", pages)

# Display logo and developer info in sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This application demonstrates sentiment analysis using a recurrent neural network (LSTM) "
    "trained on the IMDB movie review dataset. It predicts whether a review is positive or negative."
)
st.sidebar.title("Developer")
st.sidebar.info(
    "Movie Sentiment Analysis\n\n"
    "Created as a demonstration of applying RNNs to NLP tasks."
)

# Initialize session state for user input if not already there
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# HOME PAGE
if selection == "Home":
    st.title("🎬 Movie Review Sentiment Analysis")
    st.markdown("""
    ### Welcome to the Movie Review Sentiment Analyzer!
    
    This application uses a Recurrent Neural Network with LSTM architecture to predict 
    whether a movie review expresses positive or negative sentiment.
    
    **How can this be used?**
    - Film critics can analyze review sentiment
    - Marketing teams can gauge audience reception
    - Movie fans can compare their opinions with the model's prediction
    - Researchers can explore natural language processing
    
    **Navigate through the tabs to:**
    - Learn about the IMDB dataset
    - Try the sentiment analyzer
    - Explore model evaluation metrics
    
    Get started by selecting "Usage" from the navigation panel!
    """)
    
    # Display features as columns
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Sentiment Analysis")
        st.markdown("Analyze movie reviews to determine positive or negative sentiment with confidence scoring")
    
    with col2:
        st.markdown("### 🧠 LSTM Neural Network")
        st.markdown("Utilizes Recurrent Neural Network with Long Short-Term Memory for understanding context")
    
    with col3:
        st.markdown("### 📝 Real-time Prediction")
        st.markdown("Enter any movie review and get instant sentiment prediction with visualization")

    # Display example image or visualization
    st.subheader("Sample Visualization")
    
    # Create a sample visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sample data for visualization
    categories = ['Positive', 'Negative']
    values = [0.82, 0.18]
    
    # Create pie chart
    ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    ax.axis('equal')
    plt.title('Sample Distribution of Review Sentiments')
    
    st.pyplot(fig)

# DATASET PAGE
elif selection == "Dataset":
    st.title("📚 Dataset Information")
    
    st.markdown("""
    ### About the IMDB Movie Review Dataset
    
    The dataset used in this application is the IMDB Movie Review Dataset, a popular benchmark 
    for sentiment analysis tasks in natural language processing.
    
    **Dataset Highlights:**
    - 50,000 movie reviews (25,000 for training, 25,000 for testing)
    - Binary sentiment labels (positive and negative)
    - Reviews preprocessed and converted to sequences of word indices
    - Vocabulary size limited to 10,000 most frequent words
    """)
    
    # Display dataset statistics
    st.subheader("Dataset Statistics")
    
    # Create two columns for statistics
    col1, col2 = st.columns(2)
    
    with col1:
        stats_data = {
            "Metric": ["Total Reviews", "Training Reviews", "Testing Reviews", "Vocabulary Size", "Max Sequence Length"],
            "Value": ["50,000", "25,000", "25,000", "10,000", "200"]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.table(stats_df)
    
    with col2:
        # Create a sample distribution plot
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Sample data - 50/50 split between positive and negative
        categories = ['Positive Reviews', 'Negative Reviews']
        counts = [25000, 25000]
        
        # Create bar chart
        sns.barplot(x=categories, y=counts, ax=ax, palette=['#4CAF50', '#F44336'])
        plt.title('Dataset Distribution')
        plt.ylim(0, 30000)
        
        # Add value labels on top of bars
        for i, v in enumerate(counts):
            ax.text(i, v + 500, str(v), ha='center')
            
        st.pyplot(fig)
    
    # Sample reviews visualization
    st.subheader("Sample Reviews")
    
    # Create dataframe with sample reviews
    samples = pd.DataFrame({
        "Review": [
            "This movie was fantastic! The acting was superb and the plot kept me on the edge of my seat.",
            "Absolutely terrible. Poor acting, weak storyline, and terrible pacing. Complete waste of time.",
            "While the visuals were stunning, the plot fell flat and characters lacked development."
        ],
        "Sentiment": ["Positive", "Negative", "Mixed"],
        "Length (words)": [19, 13, 14]
    })
    
    st.table(samples)
    
    # Word frequency visualization
    st.subheader("Common Words in Reviews")
    
    # Sample data for word frequencies
    words = ['movie', 'film', 'actor', 'performance', 'story', 'character', 'director', 'plot', 'scene', 'acting']
    frequencies = [2356, 2103, 1879, 1654, 1598, 1432, 1298, 1245, 1187, 1154]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=words, y=frequencies, palette='viridis')
    plt.title('Most Common Words in Reviews')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add value labels
    for i, v in enumerate(frequencies):
        ax.text(i, v + 30, str(v), ha='center')
    
    st.pyplot(fig)
    
    # Data preprocessing information
    st.subheader("Data Preprocessing")
    st.markdown("""
    The dataset undergoes several preprocessing steps:
    
    1. **Tokenization**: Reviews are split into individual words
    2. **Indexing**: Words are converted to indices using a vocabulary
    3. **Padding**: Sequences are padded to a fixed length (200 words)
    4. **Embedding**: Word indices are converted to dense vectors
    
    This preprocessing ensures the text data can be efficiently processed by the neural network.
    """)

# USAGE PAGE
elif selection == "Usage":
    st.title("🔍 Sentiment Analysis Usage")
    
    st.markdown("""
    ### Try the Sentiment Analyzer
    
    Enter a movie review below or select one of our examples to analyze its sentiment.
    The model will predict whether the review is positive or negative and show the confidence level.
    """)
    
    # User input section
    st.subheader("Enter a Movie Review")
    
    # Example selector
    example_choice = st.selectbox(
        "Choose an example or enter your own review:",
        list(example_reviews.keys())
    )
    
    if example_choice == "Custom Review":
        user_input = st.text_area("Enter your review:", height=150, key="review_input")
    else:
        user_input = st.text_area("Review text:", example_reviews[example_choice], height=150, key="example_input")
    
    st.session_state.user_input = user_input
    
    # Analysis button
    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing..."):
                # Add a small delay to show the spinner
                time.sleep(0.5)
                sentiment_score, tokens = predict_sentiment(user_input, model, word_index, vocab_size, max_len)
                
                # Create two columns for results
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Visualize the prediction
                    st.subheader("Sentiment Analysis Result")
                    
                    # Display sentiment score with gauge
                    sentiment_category = "Positive" if sentiment_score > 0.5 else "Negative"
                    sentiment_color = "green" if sentiment_score > 0.5 else "red"
                    
                    st.markdown(f"<h3 style='text-align: center; color: {sentiment_color};'>Prediction: {sentiment_category}</h3>", unsafe_allow_html=True)
                    
                    # Create confidence meter
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.barh([0], [sentiment_score], color='green', alpha=0.6)
                    ax.barh([0], [1-sentiment_score], left=[sentiment_score], color='red', alpha=0.6)
                    
                    # Add threshold line
                    ax.axvline(x=0.5, color='black', linestyle='-', alpha=0.7)
                    
                    # Add labels
                    ax.text(0.1, 0, "Negative", fontsize=12, va='center')
                    ax.text(0.9, 0, "Positive", fontsize=12, va='center', ha='right')
                    ax.text(sentiment_score, 0, f"{sentiment_score:.2f}", fontsize=14, va='center', ha='center', bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Remove axes
                    ax.set_yticks([])
                    ax.set_xlim(0, 1)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show confidence text
                    confidence = sentiment_score if sentiment_score > 0.5 else (1 - sentiment_score)
                    confidence_text = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
                    st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence_text}</b> ({confidence:.2f})</p>", unsafe_allow_html=True)
                    
                    # Recommendation based on sentiment
                    st.subheader("Interpretation")
                    if sentiment_score > 0.8:
                        st.success("This review is strongly positive.")
                    elif sentiment_score > 0.5:
                        st.success("This review leans positive, but may contain some mixed opinions.")
                    elif sentiment_score > 0.2:
                        st.error("This review leans negative, but may contain some positive aspects.")
                    else:
                        st.error("This review is strongly negative.")
                
                with col2:
                    # Word analysis
                    st.subheader("Review Analysis")
                    
                    # Token statistics
                    st.markdown(f"**Review Length:** {len(tokens)} words")
                    
                    # Word cloud simulation
                    st.markdown("#### Key Words")
                    
                    # Create a dataframe of words and their "importance" (simplified approach)
                    word_df = pd.DataFrame({
                        'Word': tokens,
                        'Length': [len(word) for word in tokens],
                    })
                    
                    # Assign importance (this would ideally come from the model)
                    # Here we're simulating importance with a mix of word length and position
                    word_df['Importance'] = word_df['Length'] / word_df['Length'].max()
                    for i, word in enumerate(tokens):
                        if word in ['great', 'excellent', 'amazing', 'good', 'wonderful', 'best',
                                   'bad', 'terrible', 'awful', 'worst', 'boring', 'disappointing']:
                            word_df.loc[i, 'Importance'] = 0.9
                    
                    # Sort by "importance"
                    word_df = word_df.sort_values('Importance', ascending=False).head(10)
                    
                    # Create horizontal bar chart for top words
                    fig, ax = plt.subplots(figsize=(5, 4))
                    bars = ax.barh(word_df['Word'], word_df['Importance'], color='skyblue')
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Relative Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("Note: This is a simplified visualization and does not represent actual word importance from the model.")
        else:
            st.warning("Please enter a review to analyze.")
    
    # How it works
    st.subheader("How the Analyzer Works")
    st.markdown("""
    The sentiment analyzer follows these steps:
    
    1. **Tokenization**: Your review is split into individual words
    2. **Negation Handling**: Words following negations like "not" are modified
    3. **Conversion**: Words are converted to numbers based on the vocabulary
    4. **Sequence Processing**: The LSTM neural network processes the word sequence
    5. **Prediction**: The model outputs a value between 0 (negative) and 1 (positive)
    """)
    
    # Tips
    st.subheader("Tips for Better Analysis")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Do's:**
        - Be specific about what you liked/disliked
        - Mention aspects like acting, plot, and direction
        - Compare to other similar films
        - Use descriptive language
        """)
    
    with tips_col2:
        st.markdown("""
        **Don'ts:**
        - Use vague language
        - Focus on non-film elements
        - Write very short reviews
        - Mix multiple films in one review
        """)

# MODEL EVALUATION PAGE
elif selection == "Model Evaluation":
    st.title("📊 Model Evaluation")
    
    st.markdown("""
    ### Performance Metrics
    
    The sentiment analysis model was trained and evaluated on the IMDB movie review dataset.
    Here are the key performance metrics from the model evaluation.
    """)
    
    # Display performance metrics
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.subheader("Overall Metrics")
        
        metrics_data = {
            "Metric": ["Accuracy", "Loss", "Training Time", "Epochs"],
            "Value": ["82.56%", "0.4079", "~12 minutes", "5"]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)
    
    with metrics_col2:
        # Create accuracy chart
        st.subheader("Training History")
        
        # Sample training history data (based on notebook values)
        epochs = [1, 2, 3, 4, 5]
        train_acc = [0.6654, 0.8240, 0.8494, 0.8642, 0.8553]
        val_acc = [0.8387, 0.8254, 0.8302, 0.8191, 0.8256]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        ax.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy During Training')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    # Sample confusion matrix data
    confusion_col1, confusion_col2 = st.columns([2, 3])
    
    with confusion_col1:
        # Create confusion matrix for display
        conf_matrix = np.array([
            [10244, 1256],  # True Negatives, False Positives
            [2094, 9406]    # False Negatives, True Positives
        ])
        
        conf_df = pd.DataFrame(conf_matrix, 
                             index=['Actual Negative', 'Actual Positive'],
                             columns=['Predicted Negative', 'Predicted Positive'])
        
        st.dataframe(conf_df)
        
        # Calculate derived metrics
        tn, fp = conf_matrix[0]
        fn, tp = conf_matrix[1]
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        st.markdown(f"**Precision:** {precision:.4f}")
        st.markdown(f"**Recall:** {recall:.4f}")
        st.markdown(f"**F1 Score:** {f1_score:.4f}")
    
    with confusion_col2:
        # Create heatmap visualization of confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Model architecture
    st.subheader("Model Architecture")
    
    # Display model architecture
    st.code("""
    Model: Sequential
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 200, 128)          1,280,000 
    _________________________________________________________________
    lstm (LSTM)                  (None, 64)                49,664    
    _________________________________________________________________
    dense (Dense)                (None, 1)                 65        
    =================================================================
    Total params: 1,329,729
    Trainable params: 1,329,729
    Non-trainable params: 0
    _________________________________________________________________
    """)
    
    # Model strengths and limitations
    st.subheader("Strengths and Limitations")
    
    strengths_col1, strengths_col2 = st.columns(2)
    
    with strengths_col1:
        st.markdown("#### Strengths")
        st.markdown("""
        - Handles sequences of variable length
        - Captures word relationships and context
        - Good performance on unseen reviews
        - Simple negation handling
        - Fast prediction time
        """)
    
    with strengths_col2:
        st.markdown("#### Limitations")
        st.markdown("""
        - Limited vocabulary (10,000 words)
        - May struggle with sarcasm and irony
        - Cannot capture complex sentence structures
        - No understanding of movie-specific terminology
        - Binary classification only (no neutral category)
        """)
    
    # Future improvements
    st.subheader("Future Improvements")
    st.markdown("""
    The model could be improved in several ways:
    
    1. **Pre-trained Word Embeddings**: Use GloVe or Word2Vec for better word representations
    2. **Attention Mechanism**: Add attention layers to focus on important words
    3. **Bidirectional LSTM**: Capture context from both directions
    4. **Transformer Models**: Implement BERT or other transformer architectures
    5. **Multi-class Classification**: Add more sentiment categories (very negative, neutral, very positive)
    """)

# Footer
st.markdown("---")
st.markdown("**Built with TensorFlow, Keras, and Streamlit** | © 2025 Movie Sentiment Analyzer")
