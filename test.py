import time
import os
import sys
import PIL.Image
import OIL
import StringIO
import random

def time_test(func, arg, timelimit=10.0):
    times = []
    last_time = start_time = time.time()
    while last_time - start_time < timelimit:
        random.seed(0)
        func(arg)
        cur_time = time.time()
        times.append(cur_time - last_time)
        last_time = cur_time
    count = len(times)
    mean = sum(times) / count
    stddev = (sum([(i - mean)**2 for i in times]) / (count - 1))**0.5
    return (count, last_time - start_time, mean, stddev)

image_paths = {
    'input' : 'input.png',
    'dice' : 'dice.png',
}

images = {}

NUM_COMPOSITES = 40

def pil_composite(out):
    dest = PIL.Image.new("RGBA", (512, 512))
    for _ in xrange(NUM_COMPOSITES):
        dx = random.randint(0, 511)
        dy = random.randint(0, 511)
        dest.paste(images['dice'], (dx, dy), images['dice'])
    dest.save(out)

def oil_composite(out):
    dest = OIL.Image(512, 512)
    for _ in xrange(NUM_COMPOSITES):
        dx = random.randint(0, 511)
        dy = random.randint(0, 511)
        dest.composite(images['dice'], 255, dx, dy, 0, 0, 0, 0)
    dest.save(out)

tests = [
    ("Load", [
            ("PIL", None, lambda o: PIL.Image.open(image_paths['input']).load()),
            ("OIL", "CPU", lambda o: OIL.Image.load(image_paths['input'])),
    ]),
    
    ("Save", [
            ("PIL", None, lambda o: images['input'].save(o)),
            ("OIL", "CPU", lambda o: images['input'].save(o)),
    ]),

    ("SaveDither", [
            ("PIL", None, lambda o: images['input'].convert('RGB').convert('P', palette=PIL.Image.ADAPTIVE, colors=256).save(o)),
            ("OIL", "CPU", lambda o: images['input'].save(o, indexed=True, palette_size=256)),
    ]),
    
    ("Composite", [
            ("PIL", None, pil_composite),
            ("OIL", "CPU", oil_composite),
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
    print name + ":"
    print
    table = Table(4, 5, 17, 19, 5, 10, 5)
    table.column("", "name", "seconds per call", "standard deviation", "(%)", "file size", "(%)")
    table.column("", "----", "----------------", "------------------", "---", "---------", "---")
    bars = []
    reference_time = None
    reference_size = None
    for case, backend, func in testlist:
        if backend:
            backend_id = getattr(OIL, "BACKEND_" + backend)
            OIL.backend_set(backend_id)
            for key, path in image_paths.items():
                images[key] = OIL.Image.load(path)
        else:
            for key, path in image_paths.items():
                images[key] = PIL.Image.open(path)
        
        fname = "./test/" + case
        if backend:
            fname += " (" + backend + ")"
        fname += " " + name
        fname += ".png"
        
        _, _, percall, stddev = time_test(func, fname)
        if reference_time is None:
            reference_time = percall
        percallper = str(float(percall) / reference_time)
        percallper = percallper[:4]

        fsize = "--"
        fper = ""
        if fname:
            try:
                fsize = os.stat(fname).st_size
                if reference_size is None:
                    reference_size = fsize
                fper = float(fsize) / reference_size
            except OSError:
                pass
        table.column("", case, percall, stddev, percallper, fsize, fper)
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
