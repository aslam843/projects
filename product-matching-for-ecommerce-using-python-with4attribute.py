import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Read data
amazon = pd.read_csv("amazon_data.csv")
flipkart = pd.read_csv("flipkart_data.csv")

# Select relevant columns
amazon = amazon[['product_name', 'brand', 'Colour', 'Capacity', 'Type']]
flipkart = flipkart[['product_name', 'brand', 'Color', 'Capacity', 'Type']]

# Initialize CountVectorizer: convert text data into numerical vectors
vectorizer = CountVectorizer()

# Function to calculate cosine similarity between two texts
def calculate_similarity(vectorizer, text1, text2):
    return cosine_similarity(vectorizer.transform([text1]), vectorizer.transform([text2]))[0][0]

# Function for product name matching
def product_name_matching(vectorizer, amazon, flipkart):
    product_name_matrix = cosine_similarity(vectorizer.fit_transform(amazon['product_name'].fillna('')),
                                            vectorizer.transform(flipkart['product_name'].fillna('')))
    matching_indices = (product_name_matrix > 0.5).nonzero()
    return matching_indices, product_name_matrix

# Function for brand matching
def brand_matching(vectorizer, amazon, flipkart, matching_product_name_indices):
    matched_brands = []
    for amazon_index, flipkart_index in zip(*matching_product_name_indices):
        brand_similarity = calculate_similarity(vectorizer, amazon.iloc[amazon_index]['brand'], flipkart.iloc[flipkart_index]['brand'])
        if brand_similarity > 0.5:
            matched_brands.append((amazon_index, flipkart_index, brand_similarity))
    return matched_brands

# Function for color matching
def color_matching(vectorizer, amazon, flipkart, matched_brands):
    matched_colors = []
    for amazon_index, flipkart_index, brand_similarity in matched_brands:
        color_similarity = calculate_similarity(vectorizer, str(amazon.iloc[amazon_index]['Colour']),
                                                str(flipkart.iloc[flipkart_index]['Color']))
        if color_similarity > 0.5:
            matched_colors.append((amazon_index, flipkart_index, brand_similarity, color_similarity))
    return matched_colors

# Function for capacity matching
def capacity_matching(vectorizer, amazon, flipkart, matched_colors):
    matched_capacities = []
    for amazon_index, flipkart_index, brand_similarity, color_similarity in matched_colors:
        capacity_similarity = calculate_similarity(vectorizer, str(amazon.iloc[amazon_index]['Capacity']),
                                                str(flipkart.iloc[flipkart_index]['Capacity']))
        if capacity_similarity > 0.5:
            matched_capacities.append((amazon_index, flipkart_index, brand_similarity, color_similarity, capacity_similarity))
    return matched_capacities

# Function for capacity matching
def type_matching(vectorizer, amazon, flipkart, matched_capacities):
    matched_types = []
    for amazon_index, flipkart_index, brand_similarity, color_similarity,capacity_similarity in matched_capacities:
        type_similarity = calculate_similarity(vectorizer, str(amazon.iloc[amazon_index]['Type']),
                                                str(flipkart.iloc[flipkart_index]['Type']))
        if type_similarity > 0.5:
            matched_types.append((amazon_index, flipkart_index, brand_similarity, color_similarity, capacity_similarity, type_similarity))
    return matched_types

# Function for capacity matching
#def capacity_matching(amazon, flipkart, matched_colors):
#    matched_capacities = []
#    for amazon_index, flipkart_index, brand_similarity, color_similarity in matched_colors:
#        if amazon.iloc[amazon_index]['Capacity'] == flipkart.iloc[flipkart_index]['Capacity']:
#            matched_capacities.append((amazon_index, flipkart_index, brand_similarity, color_similarity))
#    return matched_capacities

# Function to round similarity scores
def round_similarity_score(score, decimals=2):
    return round(score, decimals)


matching_product_name_indices, product_name_matrix = product_name_matching(vectorizer, amazon, flipkart)
#print(product_name_matrix)
matched_brands = brand_matching(vectorizer, amazon, flipkart, matching_product_name_indices)
#print(matched_brands)
matched_colors = color_matching(vectorizer, amazon, flipkart, matched_brands)
#print(matched_colors)
matched_capacities = capacity_matching(vectorizer,amazon, flipkart, matched_colors)
#print(matched_capacities)
matched_types = type_matching(vectorizer,amazon, flipkart, matched_capacities)
#print(matched_types)
#matched_models = model_matching(vectorizer, amazon, flipkart, matched_capacities)


# Create result DataFrame
columns = ['Amazon_Product_Name', 'Flipkart_Product_Name','Brand_Similarity', 'Color_Similarity', 'Capacity_Similarity', 'Type_Similarity','Product_Name_Similarity']
result_df = pd.DataFrame(columns=columns)


# Fill result DataFrame
for amazon_index, flipkart_index, brand_similarity, color_similarity, capacity_similarity, type_similarity in matched_types:
    result_df = result_df.append({'Amazon_Product_Name': amazon.iloc[amazon_index]['product_name'],
                                  'Flipkart_Product_Name': flipkart.iloc[flipkart_index]['product_name'],
								  'Brand_Similarity': round_similarity_score(brand_similarity),
                                  'Color_Similarity': round_similarity_score(color_similarity),
								  'Capacity_Similarity': round_similarity_score(capacity_similarity),
								  'Type_Similarity': round_similarity_score(type_similarity),
                                  'Product_Name_Similarity': round_similarity_score(product_name_matrix[amazon_index, flipkart_index]),
                                  }, ignore_index=True)
    

# Save result DataFrame to CSV
result_df.to_csv("output.csv", index=False)

# Print final matched pairs
print("Final Matched Pairs:")
#print(result_df[columns])