class Player():
    """
    Model an NFL player, with name, position, age, overall rating 
    (from Madden 21) and average salary (from Over The Cap).
    """
    def __init__(self, name, position, age, rating, salary):
        self.name = name
        self.position = position
        self.age = age
        self.rating = rating
        self.salary = salary