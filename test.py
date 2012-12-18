import time
import os
import sys
import PIL.Image
import OIL
import StringIO
import random

def time_test(func, timelimit=10.0):
    times = []
    last_time = start_time = time.time()
    while last_time - start_time < timelimit:
        random.seed(0)
        func()
        cur_time = time.time()
        times.append(cur_time - last_time)
        last_time = cur_time
    count = len(times)
    mean = sum(times) / count
    stddev = (sum([(i - mean)**2 for i in times]) / (count - 1))**0.5
    return (count, last_time - start_time, mean, stddev)

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
            ("PIL", None, lambda: PIL.Image.open("input.png").load()),
            ("OIL", None, lambda: OIL.Image.load("input.png")),
    ]),
    
    ("Saving", [
            ("PIL", "pil.png", lambda: pilim.save("pil.png")),
            ("OIL", "oil.png", lambda: oilim.save("oil.png")),
    ]),

    ("Saving (Palette)", [
            ("PIL", "pilpalette.png", lambda: pilim.convert('RGB').convert('P', palette=PIL.Image.ADAPTIVE, colors=256).save("pilpalette.png")),
            ("OIL", "oilpalette.png", lambda: oilim.save("oilpalette.png", indexed=True, palette_size=256)),
    ]),
    
    ("Compositing", [
            ("PIL", "pilcomposite.png", pil_composite),
            ("OIL", "oilcomposite.png", oil_composite),
    ]),
]

class Table:
    def __init__(self, *columns):
        self.file = sys.stdout
        self.columns = columns
    def column(self, *entries):
        for length, text in zip(self.columns, entries):
            text = str(text)
            if len(text) > length:
                self.file.write(text[:length])
            else:
                self.file.write(text + " " * (length - len(text)))
        self.file.write("\n")

def draw_bar(offset, length, bound_low, bound_high, start, end, seq):
    file = sys.stdout
    file.write(" " * offset)
    step = (bound_high - bound_low) / length
    last_was_inside = False
    middle = (start + end) / 2
    for i in xrange(length):
        x = i * step + bound_low + (step / 2)
        if x < start or x > end:
            if x + (step / 2) > middle and x - (step / 2) <= middle:
                file.write("O")
            else:
                file.write(" ")
            last_was_inside = False
            continue
        if not last_was_inside:
            file.write("|")
            last_was_inside = True
        else:
            if x + step > end:
                file.write("|")
            elif x + (step / 2) > middle and x - (step / 2) < middle:
                file.write("O")
            else:
                file.write("-")
    file.write("\n")

for name, testlist in tests:
    table = Table(20, 5, 17, 19, 10, 5)
    table.column(name + ":", "name", "seconds per call", "standard deviation", "file size", "(%)")
    table.column("",         "----", "----------------", "------------------", "---------", "---")
    bars = []
    reference_size = None
    for case, fname, func in testlist:
        _, _, percall, stddev = time_test(func)
        fsize = "--"
        fper = ""
        if fname:
            fsize = os.stat(fname).st_size
            if reference_size is None:
                reference_size = fsize
            fper = float(fsize) / reference_size
        table.column("", case, percall, stddev, fsize, fper)
        bars.append((case, percall, stddev))
    print
    bounds = []
    for _, mean, dev in bars:
        bounds.append(mean + dev)
        bounds.append(mean - dev)
    bound_low = min(bounds)
    bound_high = max(bounds)
    for seq, mean, dev in bars:
        draw_bar(table.columns[0], sum(table.columns[1:]), bound_low, bound_high, mean - dev, mean + dev, seq)
    print
