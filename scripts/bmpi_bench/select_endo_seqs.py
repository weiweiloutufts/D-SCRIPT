import sys
# args: fasta, min length, max length, output table,  output filtered fasta


collect = False
seqbuffer = ""
name = ""
outTable = open(sys.argv[4], "w")
outFilter = open(sys.argv[5], "w")
thresh = int(sys.argv[3])
floor = int(sys.argv[2])
with open(sys.argv[1]) as fasta:
    for line in fasta:
        if line.isspace() or line[0] == "#" or not (line):
            continue
        if line[0] == ">":
            if collect:
                collect = False
                l = len(seqbuffer)
                print(name, l, sep="\t", file=outTable)
                if l >= floor and l <= thresh:
                    print(">" + name, seqbuffer, sep="\n", file=outFilter)
                seqbuffer = ""
                name = ""
            name = line.split()[0][1:]
            collect = True
        elif collect:
            seqbuffer += line.strip()
if collect:
    l = len(seqbuffer)
    print(name, l, sep="\t", file=outTable)
    if l >= floor and l <= thresh:
        print(">" + name, seqbuffer, sep="\n", file=outFilter)
outTable.close()
outFilter.close()
