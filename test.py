import time
import PIL.Image
import OIL
import StringIO

def time_test(func, timelimit=10.0):
    start_time = time.time()
    end_time = start_time
    count = 0
    while end_time - start_time < timelimit:
        func()
        count += 1
        end_time = time.time()
    return (count, end_time - start_time, (end_time - start_time) / count)

pilim = PIL.Image.open("input.png")
pilim.load()
oilim = OIL.Image.load("input.png")

tests = [
    ("Loading", [
            ("PIL", lambda: PIL.Image.open("input.png").load()),
            ("OIL", lambda: OIL.Image.load("input.png")),
    ]),
    
    ("Saving", [
            ("PIL", lambda: pilim.save("pil.png")),
            ("OIL", lambda: oilim.save("oil.png")),
    ]),

    ("Saving (Palette)", [
            ("PIL", lambda: pilim.convert('P', palette=PIL.Image.ADAPTIVE, colors=256).save("pilpalette.png")),
            ("OIL", lambda: oilim.save("oilpalette.png", indexed=True, palette_size=256)),
    ]),
]

for name, testlist in tests:
    print name
    for case, func in testlist:
        data = time_test(func)
        print "\t*", case, "\t", data
