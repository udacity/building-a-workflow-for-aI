class Fish:
    def __init__(self, color, size):
        self.color = color
        self.size = size

    def __repr__(self):
        return f"Fish(color='{self.color}', size='{self.size}')"

# List of fish attributes
fish_attributes = [
    ("red", "small"),
    ("blue", "medium"),
    ("green", "large"),
    ("yellow", "small"),
    ("orange", "medium")
]

# List to hold fish instances
fish_list = []

# Create instances in a for loop
for color, size in fish_attributes:
    fish = Fish(color, size)
    fish_list.append(fish)

# Print the list of fish instances
for fish in fish_list:
    print(fish)