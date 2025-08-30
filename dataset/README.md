# Dataset Description
This dataset is designed for personality prediction in an E-commerce setting using user-product interaction data. It is organized into three subsets: train, test, and val, each containing the following `.npy` files:

`user_features.npy`:
A NumPy array of shape (m, d₁), where m is the number of users. Each row is a one-hot encoded feature vector representing a user.

`product_features.npy`:
A NumPy array of shape (n, d₂), where n is the number of products. Each row is a one-hot encoded feature vector representing a product.

`user_product.npy`:
A NumPy array of shape (e, 2), representing e user-product interactions. Each row contains:

Column 0: `user_id`

Column 1: `product_id`

`label.npy`:
A NumPy array of shape (m, 9), where each row is a multi-label binary vector representing the personalities assigned to a user. Each column corresponds to one personality class.

Persona Label Mapping (for `label.npy`)
Column Index	Personality Class
0	`fashion_enthusiast_label`
1	`budget_shopper_label`
2	`sport_shoppers_label`
3	`luxury_shoppers_label`
4	`professional_attire_shoppers_label`
5	`casual/comfort_shoppers_label`
6	`adventure_shoppers_label`
7	`children_clothing_shoppers_label`
8	`wedding_wear_shoppers_label`
