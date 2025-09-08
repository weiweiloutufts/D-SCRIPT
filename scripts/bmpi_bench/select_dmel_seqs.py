import sys
# args: isoform table, fasta, output table, output fasta, max length, output filtered fasta

prots = set()
genes = set()

with open(sys.argv[1]) as isoforms:
    for line in isoforms:
        if line.isspace() or line[0] == "#" or not (line):
            continue
        tokens = line.strip().split()
        gene = tokens[0]
        if gene not in genes:
            genes.add(gene)
            prots.add(tokens[2])

collect = False
seqbuffer = ""
id = ""
name = ""
outTable = open(sys.argv[3], "w")
outFasta = open(sys.argv[4], "w")
if len(sys.argv) > 5:
    filter = True
    thresh = int(sys.argv[5])
    outFilter = open(sys.argv[6], "w")
else:
    filer = False

with open(sys.argv[2]) as fasta:
    for line in fasta:
        if line.isspace() or line[0] == "#" or not (line):
            continue
        if line[0] == ">":
            if collect:
                collect = False
                l = len(seqbuffer)
                print(id, name, l, sep="\t", file=outTable)
                print(">" + id, seqbuffer, sep="\n", file=outFasta)
                if filter and l <= thresh:
                    print(">" + id, seqbuffer, sep="\n", file=outFilter)
                seqbuffer = ""
                id = ""
                name = ""
            tokens = line.split(";")
            id = tokens[0].split()[0][1:]
            name = ""
            for token in tokens[1:]:
                pair = token.split("=")
                if pair[0].strip() == "name":
                    name = pair[1]
                    break
            if name and name in prots:
                collect = True
        elif collect:
            seqbuffer += line.strip()
if collect:
    l = len(seqbuffer)
    print(id, name, l, sep="\t", file=outTable)
    print(">" + id, seqbuffer, sep="\n", file=outFasta)
    if filter and l <= thresh:
        print(">" + id, seqbuffer, sep="\n", file=outFilter)
outTable.close()
outFasta.close()
outFilter.close()
