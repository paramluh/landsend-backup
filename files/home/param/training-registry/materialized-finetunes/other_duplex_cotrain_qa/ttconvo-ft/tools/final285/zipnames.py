import sys, zipfile
for p in sys.argv[1:]:
    z = zipfile.ZipFile(p)
    n = z.namelist()
    print(p, len(n), n[:3], sum(len(x) for x in n))
