import random
import sys

f = open(sys.argv[1])
prots = [x.strip() for x in f.readlines()]
pairs = [(p, q) for i, p in enumerate(prots) for q in prots[i + 1 :]]
np = len(pairs)
target = int(float(sys.argv[2]) * np)
print(f"Choosing {target} pairs from {np}", file=sys.stderr)
random.seed(0)
sel = random.sample(range(np), k=target)
for i in sel:
    print(*pairs[i], sep="\t")
