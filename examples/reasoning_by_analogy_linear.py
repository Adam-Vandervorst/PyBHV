# The linear variant of https://colab.research.google.com/drive/10XSpooxDAeYVMivF3W2-N0W5yEIUfdZl#scrollTo=dQ7044izy3_y
# This is of theoretical interest
from bhv.vanilla import VanillaBHV as BHV
from bhv.variants import LinearBHVModel


# Operation on linear are checked
linear = LinearBHVModel(BHV)

# declare up front how many spaces and entropy we're going to use
# ONE and ZERO are singletons, so using permute(0) to copy them for linearity checking
symbol_ones = [BHV.ONE.permute(0) for _ in range(9)]
symbol_zeros = [BHV.ZERO.permute(0) for _ in range(9)]
copy_ones = [BHV.ONE.permute(0) for _ in range(6)]
copy_zeros = [BHV.ZERO.permute(0) for _ in range(6)]


((name, neg_name), (capital_city, neg_capital_city), (money, neg_money),
 (united_states, _neg), (washington_dc, _neg), (dollar, neg_dollar),
 (mexico, _neg), (mexico_city, neg_mexico_city), (peso, _neg)) = [linear.spawn(one, zero)
    for one, zero in zip(symbol_ones, symbol_zeros)]

(name_is_united_states, _and, _and) = linear.xor(name, united_states)
(capital_city_is_washington_dc, _and, _and) = linear.xor(capital_city, washington_dc)
(money_is_dollar, _and, _and) = linear.xor(money, dollar)

(_ands, USA, _ors) = linear.thresholds3(name_is_united_states, capital_city_is_washington_dc, money_is_dollar)

# note to make more than on record, we need to re-use the field names
# we invert the negated version we got during spawning the random vectors to obtain copies
((name_copy, _id, _id),
 (capital_city_copy, _id, _id),
 (money_copy, _id, _id),
 (dollar_copy, _id, _id),
 (mexico_city_copy, _id, _id)) = [linear.invert(neg, one, zero)
    for neg, one, zero in zip([neg_name, neg_capital_city, neg_money, neg_dollar, neg_mexico_city], copy_ones, copy_zeros)]

(name_is_mexico, _and, _and) = linear.xor(name_copy, mexico)
(capital_city_is_mexico_city, _and, _and) = linear.xor(capital_city_copy, mexico_city)
(money_is_peso, _and, _and) = linear.xor(money_copy, peso)

(_ands, MEX, _ors) = linear.thresholds3(name_is_mexico, capital_city_is_mexico_city, money_is_peso)

(orig_pair, _and, _and) = linear.xor(USA, MEX)
# another method of copying
(_neg, pair_one, pair_two) = linear.invert(orig_pair, copy_ones[-1], copy_zeros[-1])

(dollar_of_mexico, _and, _and) = linear.xor(dollar_copy, pair_one)

# ignoring linearity for these checks, we'd have to make another copy of peso and dollar
assert not peso.unrelated(dollar_of_mexico)
assert peso.hamming(dollar_of_mexico) < dollar.hamming(dollar_of_mexico)

(mexico_city_of_usa, _and, _and) = linear.xor(mexico_city_copy, pair_two)

# ignoring linearity
assert not washington_dc.unrelated(mexico_city_of_usa)
assert washington_dc.hamming(mexico_city_of_usa) < mexico_city.hamming(mexico_city_of_usa)
