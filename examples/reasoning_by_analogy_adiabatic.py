# The adiabatic variant of https://colab.research.google.com/drive/10XSpooxDAeYVMivF3W2-N0W5yEIUfdZl#scrollTo=dQ7044izy3_y
# This is of theoretical interest
from bhv.vanilla import VanillaBHV as BHV, DIMENSION
from bhv.variants import AdiabaticBHVModel, Tank


# declare up front how much charge we have in our tank
tank = Tank(10*DIMENSION, record=True)
adiabatic = AdiabaticBHVModel(tank, BHV)

((name, neg_name), (capital_city, neg_capital_city), (money, neg_money),
 (united_states, _neg), (washington_dc, _neg), (dollar, neg_dollar),
 (mexico, _neg), (mexico_city, neg_mexico_city), (peso, _neg)) = [adiabatic.spawn() for _ in range(9)]

name_is_united_states = adiabatic.xor(name, united_states)
capital_city_is_washington_dc = adiabatic.xor(capital_city, washington_dc)
money_is_dollar = adiabatic.xor(money, dollar)

# majority is quite a flooding operation
USA = adiabatic.majority3(name_is_united_states, capital_city_is_washington_dc, money_is_dollar)

# note to make more than on record, we need to re-use the field names
# we invert the negated version we got during spawning the random vectors to obtain copies to save charge
(name_copy,
 capital_city_copy,
 money_copy,
 dollar_copy,
 mexico_city_copy) = [adiabatic.invert(neg)
    for neg in [neg_name, neg_capital_city, neg_money, neg_dollar, neg_mexico_city]]

name_is_mexico = adiabatic.xor(name_copy, mexico)
capital_city_is_mexico_city = adiabatic.xor(capital_city_copy, mexico_city)
money_is_peso = adiabatic.xor(money_copy, peso)

MEX = adiabatic.majority3(name_is_mexico, capital_city_is_mexico_city, money_is_peso)

orig_pair = adiabatic.xor(USA, MEX)
# another method of copying
(pair_one, pair_two, _neg) = adiabatic.switch(orig_pair, adiabatic.get_one(), adiabatic.get_zero())

dollar_of_mexico = adiabatic.xor(dollar_copy, pair_one)

# ignoring adiabaticity for these checks, we'd have to make another copy of peso and dollar
assert not peso.unrelated(dollar_of_mexico)
assert peso.hamming(dollar_of_mexico) < dollar.hamming(dollar_of_mexico)

mexico_city_of_usa = adiabatic.xor(mexico_city_copy, pair_two)

# ignoring adiabaticity
assert not washington_dc.unrelated(mexico_city_of_usa)
assert washington_dc.hamming(mexico_city_of_usa) < mexico_city.hamming(mexico_city_of_usa)

historical_levels = tank.historical_levels()

print("final tank level:", tank.level, "historical min,max:", min(historical_levels), max(historical_levels))
