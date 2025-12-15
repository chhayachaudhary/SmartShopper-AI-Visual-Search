import requests
from bs4 import BeautifulSoup
import random
import time
import re 
import streamlit as st 

# --- Large Pool of Real-World-Style Reviews ---
# This list simulates retrieving reviews from a live API/Database query.
REVIEW_POOL = {
    # POSITIVE/HIGH POLARITY REVIEWS
    "positive": [
        "Absolutely amazing quality for the price! Exceeded all expectations. Highly recommend.",
        "The best purchase I've made all year. The fit is perfect, and the material feels premium.",
        "Five stars! Incredible value. It arrived quickly and was exactly as described.",
        "This product is fantastic! Solved my problem instantly and looks stylish too. Excellent buy.",
        "Highly satisfied! The fabric is soft, durable, and the color is vibrant.",
        "Worth every single penny. I'd recommend this to anyone looking for quality.",
        "Extremely happy with this item. It's truly a game-changer.",
    ],
    # NEGATIVE/LOW POLARITY REVIEWS
    "negative": [
        "Very poor quality. It looks cheap and feels flimsy. A complete waste of money.",
        "Extremely disappointed. The item arrived damaged and the sizing was completely wrong.",
        "I regret buying this. The material faded immediately after the first wash.",
        "Customer service was terrible, and the product broke within a week. Avoid this brand.",
        "Not satisfied at all. It looked much better in the pictures online.",
        "Overpriced for what it is. The experience was frustrating and slow.",
        "I would give zero stars if I could. This was the worst online purchase ever.",
    ],
    # NEUTRAL/MIXED REVIEWS
    "neutral": [
        "It's okay for the price, nothing special. Does the job.",
        "The shipping was slow, but the product itself is decent.",
        "Looks a bit different than the picture, but I'll keep it.",
        "Average quality. Nothing to complain about, but nothing exciting either.",
    ]
}


def generate_dynamic_reviews(quality_bias, num_reviews=6):
    """Generates a list of reviews with sentiment biased by the platform's 'quality_bias'."""
    reviews = []
    
    # Calculate how many positive and negative reviews to pull based on the quality bias (0 to 1)
    # Higher bias means more positive reviews
    num_positive = int(num_reviews * (0.4 + quality_bias * 0.3)) + random.randint(0, 1)
    num_negative = num_reviews - num_positive
    
    # Ensure limits are respected
    num_positive = max(2, min(num_positive, len(REVIEW_POOL["positive"])))
    num_negative = max(0, min(num_negative, len(REVIEW_POOL["negative"])))
    num_neutral = num_reviews - num_positive - num_negative
    
    # Collect reviews
    reviews.extend(random.sample(REVIEW_POOL["positive"], num_positive))
    reviews.extend(random.sample(REVIEW_POOL["negative"], num_negative))
    reviews.extend(random.sample(REVIEW_POOL["neutral"], min(num_neutral, len(REVIEW_POOL["neutral"]))))
    
    random.shuffle(reviews)
    return reviews


def get_mock_data(product_name):
    """
    Generates structured, dynamically correlated mock data.
    """
    search_query_url = product_name.replace(' ', '+')
    
    # Define hidden quality bias and base price variance for each mock platform
    # This makes the results look like real competitive data
    platform_data_structure = [
        {"platform": "Amazon", "base_price_factor": 1.1, "quality_bias": 0.9, "url_pattern": "https://www.amazon.in/s?k="}, # High Quality, High Price
        {"platform": "Flipkart", "base_price_factor": 1.0, "quality_bias": 0.6, "url_pattern": "https://www.flipkart.com/search?q="}, # Medium Quality, Medium Price
        {"platform": "Meesho", "base_price_factor": 0.8, "quality_bias": 0.4, "url_pattern": "https://www.meesho.com/search?q="}, # Low Quality, Low Price
    ]

    base_price = random.randint(1000, 3000) # Establish a base price for the product category
    
    final_data = []

    for p_data in platform_data_structure:
        # 1. Price/Discount Correlation
        price = int(base_price * p_data['base_price_factor'] * random.uniform(0.9, 1.1)) # Apply variance
        original_price = int(price / (1 - random.uniform(0.1, 0.5))) # Ensure a believable original price
        if original_price <= price: original_price = price + random.randint(100, 500) # Safety check

        # 2. Rating Correlation (Higher bias = higher rating)
        rating = round(random.uniform(3.0, 4.0) + p_data['quality_bias'] * 0.5, 1)
        rating = min(rating, 5.0) # Cap rating at 5.0
        
        # 3. Dynamic Reviews (Biased by quality)
        reviews = generate_dynamic_reviews(p_data['quality_bias'])
        
        final_data.append({
            "platform": p_data['platform'],
            "title": f"{product_name} - {p_data['platform']} Edition",
            "price": price,
            "original_price": original_price,
            "rating": rating,
            "reviews": reviews,
            "url": f"{p_data['url_pattern']}{search_query_url}"
        })
    
    # Calculate discount percentage
    for item in final_data:
        item['discount'] = int(((item['original_price'] - item['price']) / item['original_price']) * 100)
        
    return final_data

def is_url(text):
    """Checks if the input string is likely a URL using a simple regex pattern."""
    return re.match(r'^https?://', text) is not None

def extract_product_name_from_url(url):
    """
    SIMULATED FUNCTION: Extracts a plausible product name from the URL path.
    """
    clean_url = re.sub(r'^https?://', '', url).split('?')[0]
    path_segments = [s for s in re.split(r'[/\-.]', clean_url) if s and not re.match(r'\d+', s)]
    
    if path_segments:
        name = " ".join(path_segments[-2:]).title()
        return name
    
    return "Generic Product Item"

def search_products(query):
    """
    Main search handler. Checks for URL input, extracts name, and returns mock data.
    """
    product_name = query.strip()
    
    if is_url(product_name):
        time.sleep(3) # Simulate longer scraping/API delay
        product_name = extract_product_name_from_url(product_name)
        st.success(f"URL Scanned! Searching for: **{product_name}**")
    else:
        time.sleep(2) # Simulate normal text search delay

    return get_mock_data(product_name)