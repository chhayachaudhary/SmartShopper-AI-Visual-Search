from textblob import TextBlob

def analyze_sentiment(reviews):
    """
    Analyzes a list of review strings using TextBlob (a simple NLP tool)
    and returns an aggregated sentiment score and verbal verdict.
    """
    if not reviews:
        # Return neutral values if there are no reviews to prevent division by zero
        return 0, "No Reviews"
    
    total_polarity = 0
    # TextBlob analysis loop
    for review in reviews:
        # TextBlob's polarity attribute is a float between -1.0 (negative) and +1.0 (positive)
        analysis = TextBlob(review)
        total_polarity += analysis.sentiment.polarity
    
    # Calculate the average sentiment score
    avg_polarity = total_polarity / len(reviews)
    
    # Determine the verbal verdict based on the average score
    if avg_polarity > 0.5:
        verdict = "Excellent 😍"
    elif avg_polarity > 0.1:
        verdict = "Good 🙂"
    elif avg_polarity >= -0.1: # Neutral range
        verdict = "Neutral 😐"
    else:
        verdict = "Negative 😠"
        
    return round(avg_polarity, 2), verdict