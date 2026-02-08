"""Test classifier blueprints in various container contexts."""

from abc import ABC

from cyantic import CyanticModel, classifier


class Animal(ABC, CyanticModel):
    """Base animal class."""

    name: str
    age: int

    def speak(self) -> str:
        """Make animal sound."""
        raise NotImplementedError


class Dog(Animal):
    """Dog implementation."""

    breed: str

    def speak(self) -> str:
        return f"Woof! I'm {self.name}, a {self.breed}"


class Cat(Animal):
    """Cat implementation."""

    color: str

    def speak(self) -> str:
        return f"Meow! I'm {self.name}, a {self.color} cat"


# Register classifier for Animal
classifier(Animal)


class Zoo(CyanticModel):
    """Zoo containing animals in various container types."""

    name: str
    animals: list[Animal]
    featured_animals: dict[str, Animal]


def test_classifier_direct_build():
    """Test that classifier works for direct model building."""
    config = {
        "cls": "test_classifier_containers.Dog",
        "kwargs": {
            "name": "Buddy",
            "age": 5,
            "breed": "Golden Retriever"
        }
    }

    animal = Animal.build(config)
    assert isinstance(animal, Dog)
    assert animal.name == "Buddy"
    assert animal.age == 5
    assert animal.breed == "Golden Retriever"
    assert animal.speak() == "Woof! I'm Buddy, a Golden Retriever"


def test_classifier_in_list():
    """Test that classifier works for items in lists."""
    config = {
        "name": "Safari Park",
        "animals": [
            {
                "cls": "test_classifier_containers.Dog",
                "kwargs": {
                    "name": "Buddy",
                    "age": 5,
                    "breed": "Golden Retriever"
                }
            },
            {
                "cls": "test_classifier_containers.Cat",
                "kwargs": {
                    "name": "Whiskers",
                    "age": 3,
                    "color": "orange"
                }
            }
        ],
        "featured_animals": {
            "star": {
                "cls": "test_classifier_containers.Dog",
                "kwargs": {
                    "name": "Rex",
                    "age": 7,
                    "breed": "German Shepherd"
                }
            }
        }
    }

    zoo = Zoo.build(config)
    
    # Check zoo name
    assert zoo.name == "Safari Park"
    
    # Check list animals
    assert len(zoo.animals) == 2
    
    buddy = zoo.animals[0]
    assert isinstance(buddy, Dog)
    assert buddy.name == "Buddy"
    assert buddy.breed == "Golden Retriever"
    
    whiskers = zoo.animals[1]
    assert isinstance(whiskers, Cat)
    assert whiskers.name == "Whiskers"
    assert whiskers.color == "orange"
    
    # Check dict animals
    rex = zoo.featured_animals["star"]
    assert isinstance(rex, Dog)
    assert rex.name == "Rex"
    assert rex.breed == "German Shepherd"


def test_classifier_mixed_with_regular_fields():
    """Test classifier alongside regular model fields."""
    
    class Park(CyanticModel):
        """Park with both regular and classifier fields."""
        
        location: str
        capacity: int
        main_attraction: Animal
        backup_animals: list[Animal]
    
    config = {
        "location": "Downtown",
        "capacity": 100,
        "main_attraction": {
            "cls": "test_classifier_containers.Cat",
            "kwargs": {
                "name": "Fluffy",
                "age": 2,
                "color": "white"
            }
        },
        "backup_animals": [
            {
                "cls": "test_classifier_containers.Dog",
                "kwargs": {
                    "name": "Spot",
                    "age": 4,
                    "breed": "Dalmatian"
                }
            }
        ]
    }
    
    park = Park.build(config)
    
    # Check regular fields
    assert park.location == "Downtown"
    assert park.capacity == 100
    
    # Check classifier fields
    assert isinstance(park.main_attraction, Cat)
    assert park.main_attraction.name == "Fluffy"
    
    assert len(park.backup_animals) == 1
    assert isinstance(park.backup_animals[0], Dog)
    assert park.backup_animals[0].name == "Spot"