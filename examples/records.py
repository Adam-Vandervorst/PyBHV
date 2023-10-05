from typing import Optional
from dataclasses import dataclass

from bhv.embedding import Intervals, Random
from bhv.vanilla import VanillaBHV as BHV


@dataclass
class Person:
    name: str
    age: float
    pet: 'Cat | Dog'

@dataclass
class Cat:
    name: str
    age: float

@dataclass
class Dog:
    name: str
    age: float


Person_hv, Cat_hv, Dog_hv = BHV.nrand(3)
name_embed = Random(BHV)
age_embed = Intervals.perfect(BHV, 0, 100, 200, 10)

def encode_person(p: Person) -> BHV:
    name_hv = name_embed.forward(p.name).permute(1)
    age_hv = age_embed.forward(p.age).permute(1)

    if isinstance(p.pet, Cat):
        cat_name_hv = name_embed.forward(p.pet.name).permute(2)
        cat_age_hv = age_embed.forward(p.pet.age).permute(2)
        pet_hv = BHV.majority([Cat_hv, cat_name_hv, cat_age_hv]).permute(1)
    else:
        dog_name_hv = name_embed.forward(p.pet.name).permute(3)
        dog_age_hv = age_embed.forward(p.pet.age).permute(3)
        pet_hv = BHV.majority([Dog_hv, dog_name_hv, dog_age_hv]).permute(1)

    return BHV.majority([Person_hv, name_hv, age_hv, pet_hv])

def decode_person(hv: BHV) -> Person:
    assert not hv.unrelated(Person_hv)
    person_hv = hv.permute(-1)
    name = name_embed.back(person_hv)
    age = age_embed.back(person_hv)
    if not person_hv.unrelated(Cat_hv):
        cat_hv = person_hv.permute(-2)
        cat_name = name_embed.back(cat_hv)
        cat_age = age_embed.back(cat_hv)
        pet = Cat(cat_name, cat_age)
    elif not person_hv.unrelated(Dog_hv):
        dog_hv = person_hv.permute(-3)
        dog_name = name_embed.back(dog_hv)
        dog_age = age_embed.back(dog_hv)
        pet = Dog(dog_name, dog_age)
    else:
        assert False

    return Person(name, age, pet)


if __name__ == '__main__':
    joe = Person("Joe", 16.5, Dog("Blacky", 1))
    mia = Person("Mia", 61, Cat("Lucy", 9))

    joe_hv = encode_person(joe)
    mia_hv = encode_person(mia)

    print(decode_person(joe_hv))
    print(decode_person(mia_hv))
