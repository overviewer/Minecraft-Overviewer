import time
import PIL.Image
import OIL
import StringIO
import random

def time_test(func, timelimit=10.0):
    start_time = time.time()
    end_time = start_time
    count = 0
    while end_time - start_time < timelimit:
        random.seed(0)
        func()
        count += 1
        end_time = time.time()
    return (count, end_time - start_time, (end_time - start_time) / count)

pilim = PIL.Image.open("input.png")
pilim.load()
oilim = OIL.Image.load("input.png")

pildice = PIL.Image.open("dice.png")
pildice.load()
oildice = OIL.Image.load("dice.png")

NUM_COMPOSITES = 40

def pil_composite():
    dest = PIL.Image.new("RGBA", (512, 512))
    for _ in xrange(NUM_COMPOSITES):
        dx = random.randint(0, 511)
        dy = random.randint(0, 511)
        dest.paste(pildice, (dx, dy), pildice)
    dest.save("pilcomposite.png")

def oil_composite():
    dest = OIL.Image(512, 512)
    for _ in xrange(NUM_COMPOSITES):
        dx = random.randint(0, 511)
        dy = random.randint(0, 511)
        dest.composite(oildice, 255, dx, dy, 0, 0, 0, 0)
    dest.save("oilcomposite.png")

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
            ("PIL", lambda: pilim.convert('RGB').convert('P', palette=PIL.Image.ADAPTIVE, colors=256).save("pilpalette.png")),
            ("OIL", lambda: oilim.save("oilpalette.png", indexed=True, palette_size=256)),
    ]),
    
    ("Compositing", [
            ("PIL", pil_composite),
            ("OIL", oil_composite),
    ]),
]

for name, testlist in tests:
    print name
    for case, func in testlist:
        data = time_test(func)
        print "\t*", case, "\t", data
