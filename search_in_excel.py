import pandas as pd

# Load the two Excel files
file1 = 'file11.xlsx'  # The file containing the search values
file2 = 'file12.xlsx'  # The file to be searched

# Read the Excel files into DataFrames
df1 = pd.read_excel(file1, sheet_name='file11')  # Adjust sheet name if needed
df2 = pd.read_excel(file2, sheet_name='file12')  # Adjust sheet name if needed

# Specify the columns to search and to be searched
search_column = 'product_name'  # Column in file1 to search
target_column = 'product_name'  # Column in file2 to search against

# Perform the search
search_values = df1[search_column].tolist()
results = df2[df2[target_column].isin(search_values)]

# Display the results
print(results)

# Optionally, save the results to a new Excel file
results.to_excel('search_results.xlsx', index=False)