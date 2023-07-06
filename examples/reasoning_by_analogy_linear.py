# The linear variant of https://colab.research.google.com/drive/10XSpooxDAeYVMivF3W2-N0W5yEIUfdZl#scrollTo=dQ7044izy3_y
# This is purely of theoretical interest
from bhv.vanilla import VanillaBHV as BHV
from bhv.variants import LinearBHV


# declare up front how many spaces and entropy we're going to use
symbol_ones = [BHV.ONE]*9
symbol_zeros = [BHV.ZERO]*9
copy_ones = [BHV.ONE]*3
copy_zeros = [BHV.ZERO]*3


((name, neg_name), (capital_city, neg_capital_city), (money, neg_money),
 (united_states, _neg), (washington_dc, _neg), (dollar, _neg),
 (mexico, _neg), (mexico_city, _neg), (peso, _neg)) = [LinearBHV.spawn(BHV, one, zero)
                                                       for one, zero in zip(symbol_ones, symbol_zeros)]

(name_is_united_states, _and, _and) = LinearBHV.xor(name, united_states)
(capital_city_is_washington_dc, _and, _and) = LinearBHV.xor(capital_city, washington_dc)
(money_is_dollar, _and, _and) = LinearBHV.xor(money, dollar)

(_ands, USA, _ors) = LinearBHV.thresholds3(BHV, name_is_united_states, capital_city_is_washington_dc, money_is_dollar)

# note to make more than on record, we need to re-use the field names
# we invert the negated version we got during spawning the random vectors to obtain copies
((name_copy, _id, _id),
 (capital_city_copy, _id, _id),
 (money_copy, _id, _id)) = [LinearBHV.invert(neg, one, zero)
                            for neg, one, zero in zip([neg_name, neg_capital_city, neg_money], copy_ones, copy_zeros)]

(name_is_mexico, _and, _and) = LinearBHV.xor(name_copy, mexico)
(capital_city_is_mexico_city, _and, _and) = LinearBHV.xor(capital_city_copy, mexico_city)
(money_is_peso, _and, _and) = LinearBHV.xor(money_copy, peso)

(_ands, MEX, _ors) = LinearBHV.thresholds3(BHV, name_is_mexico, capital_city_is_mexico_city, money_is_peso)

(Pair, _and, _and) = LinearBHV.xor(USA, MEX)


(dollar_of_mexico, _and, _and) = LinearBHV.xor(dollar, Pair)

# ignoring linearity for these checks, we'd have to make another copy of peso and dollar
assert not peso.unrelated(dollar_of_mexico)
assert peso.hamming(dollar_of_mexico) < dollar.hamming(dollar_of_mexico)

(mexico_city_of_usa, _and, _and) = LinearBHV.xor(mexico_city, Pair)

# ignoring linearity
assert not washington_dc.unrelated(mexico_city_of_usa)
assert washington_dc.hamming(mexico_city_of_usa) < mexico_city.hamming(mexico_city_of_usa)
