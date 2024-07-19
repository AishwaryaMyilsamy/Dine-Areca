from collections import defaultdict
import matplotlib.pyplot as plt

# Initialize counters for each product type
product1_count = 0
product2_count = 0
product3_count = 0
product4_count = 0
product5_count = 0

# Function to increase count for a selected product
def increase_product_count(product):
    global product1_count, product2_count, product3_count, product4_count, product5_count
    if product == "Product 1":
        product1_count += 1
    elif product == "Product 2":
        product2_count += 1
    elif product == "Product 3":
        product3_count += 1
    elif product == "Product 4":
        product4_count += 1
    elif product == "Product 5":
        product5_count += 1

# Example usage: Increase count for different products
increase_product_count("Product 1")
increase_product_count("Product 2")
increase_product_count("Product 1")
increase_product_count("Product 3")
increase_product_count("Product 1")
increase_product_count("Product 4")
increase_product_count("Product 2")
# Increment counts for other products as needed...

# Print individual counts for each product
print(f"Product 1 Count: {product1_count}")
print(f"Product 2 Count: {product2_count}")
print(f"Product 3 Count: {product3_count}")
print(f"Product 4 Count: {product4_count}")
print(f"Product 5 Count: {product5_count}")

# Calculate total count of all products
total_products = product1_count + product2_count + product3_count + product4_count + product5_count
print(f"Total products: {total_products}")

# Generate pie chart
labels = ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5']
sizes = [product1_count, product2_count, product3_count, product4_count, product5_count]
colors = ['#8A2BE2', '#9370DB', '#A020F0', '#BA55D3', '#DA70D6']  # Purple gradients

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Products', fontsize=16)
plt.axis('equal')
plt.show()
