# Let's try and encode some rules, and do some rule-based computing
# If x is the mother of y and y is the father of z then x is the grandmother of z

# from bhv.np import NumPyPacked64BHV as BHV, NumPyWordPermutation as Perm
from bhv.vanilla import VanillaBHV as BHV, VanillaPermutation as Perm

# relation utility
rel_subject = Perm.random()
rel_object = Perm.random()

# relations
mother_of = BHV.rand()
father_of = BHV.rand()
grandmother_of = BHV.rand()

def apply_rel(rel, x, y):
  sx = rel_subject(rel) ^ x
  sy = rel_object(rel) ^ y
  return BHV.majority([sx, sy])

# our rule, read `xor` as "implied by" and `BHV.majority` as "and"
# note this is applied to multiple "datapoints" ...
def generate_sample():
  person_x = BHV.rand()
  person_y = BHV.rand()
  person_z = BHV.rand()

  mxy = apply_rel(mother_of, person_x, person_y)
  fyz = apply_rel(father_of, person_y, person_z)
  gxz = apply_rel(grandmother_of, person_x, person_z)

  return gxz ^ BHV.majority([mxy, fyz])

# ... and averaged out for higher accuracy
grandmother_rule = BHV.majority([generate_sample() for _ in range(15)])

# applying grandmother rule
anna = BHV.rand()
bill = BHV.rand()
cid = BHV.rand()
anna_mother_of_bill = apply_rel(mother_of, anna, bill)
bill_father_of_cid = apply_rel(father_of, bill, cid)
calculated_anna_grandmother_of_cid = grandmother_rule ^ BHV.majority([anna_mother_of_bill, bill_father_of_cid])
actual_anna_grandmother_of_cid = apply_rel(grandmother_of, anna, cid)

assert calculated_anna_grandmother_of_cid.related(actual_anna_grandmother_of_cid)
