"""How do v1 and v2 final lists differ: same set reordered vs different programmes.
(Same code as run inline in the session; output in 97_v1_v2_diff.log.)"""
import pandas as pd
a = pd.read_csv("out/submission_lr_fusion.csv"); b = pd.read_csv("out/submission_lr_fusion_v2.csv")
assert (a.user_id == b.user_id).all()
la, lb = a.tv_show_id.str.split(), b.tv_show_id.str.split()
diff = la != lb
same_set = [set(x) == set(y) for x, y in zip(la, lb)]
n_new = [len(set(y) - set(x)) for x, y in zip(la, lb)]
d = pd.DataFrame({"diff": diff, "same_set": same_set, "n_new": n_new})
print("users:", len(d), "| lists differ:", int(d["diff"].sum()))
print("of differing: same 5 programmes, only order changed:", int((d["diff"] & d.same_set).sum()),
      "| at least one programme replaced:", int((d["diff"] & ~d.same_set).sum()))
print("replaced programmes per differing user:", d[d["diff"]].n_new.value_counts().sort_index().to_dict())
